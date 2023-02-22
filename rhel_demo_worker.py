#!/usr/bin/env python
import os
import io
import pika
import gridfs
from pymongo import MongoClient
from PIL import Image
import requests
import json

import numpy as np
from bson.objectid import ObjectId

class RHELDemoWorker:
    def __init__( self ):
        self.rabbitmq_host = os.getenv( 'RABBITMQ_HOST' )
        self.rabbitmq_username = os.getenv( 'RABBITMQ_USERNAME' )
        self.rabbitmq_password = os.getenv( 'RABBITMQ_PASSWORD' )
        self.rabbitmq_queue = os.getenv( 'RABBITMQ_QUEUE' )

        self.mongodb_connection_string = os.getenv( 'MONGODB_CONNECTION_STRING' )
        self.mongodb_host = os.getenv( 'MONGODB_HOST' )
        self.mongodb_port = os.getenv( 'MONGODB_PORT', 27017 )
        self.mongodb_username = os.getenv( 'MONGODB_USERNAME' )
        self.mongodb_password = os.getenv( 'MONGODB_PASSWORD' )
        self.mongodb_db = os.getenv( 'MONGODB_DB' )
        self.mongodb_auth_type = os.getenv( 'MONGODB_AUTH_MECHNISM', 'DEFAULT' )

        self.server_url = os.getenv( 'SERVER_URL' )
        self.top_x = os.getenv( 'TOP_X', 5 )

    def connect_rabbitmq( self ):
        credentials = pika.credentials.PlainCredentials(self.rabbitmq_username, self.rabbitmq_password)
        hosts = []
        for host in self.rabbitmq_host.split(','):
          hosts.append( pika.ConnectionParameters( host.strip(), credentials=credentials ) )
        connection = pika.BlockingConnection( hosts )
        return connection.channel()

    def connect_mongodb( self ):
        if self.mongodb_connection_string:
          client = MongoClient( self.mongodb_connection_string )
        else:
          client = MongoClient(
                self.mongodb_host,
                self.mongodb_port,
                username=self.mongodb_username,
                password=self.mongodb_password,
                authMechanism=self.mongodb_auth_type
            )
        return client[ self.mongodb_db ]

    def rabbitmq_callback( self, ch, method, properties, body):
        print(" [x] Received %r" % body)
        self.process_image( ObjectId(body.decode('utf-8')) )
        #self.process_image( body )
        ch.basic_ack( delivery_tag = method.delivery_tag )

    def process_image( self, item_id ):
        '''
        with open( '/tmp/image', 'rb') as f:
            tmpimage = f.read()
            db = self.connect_mongodb()
            fs = gridfs.GridFS(db)
            item_id = fs.put(tmpimage)
        '''
        image = self.retreive_image( item_id )
        image = self.compress_image( item_id, image)
        prediction = self.classify_image( image )
        prediction = self.convert_prediction( prediction )
        self.tag_predictions( item_id, prediction )

    def retreive_image( self, item_id ):
        db = self.connect_mongodb()
        fs = gridfs.GridFS(db)
        return fs.get( item_id ).read()

    def save_image( self, item_id, image ):
        db = self.connect_mongodb()
        fs = gridfs.GridFS(db)
        fs.delete(item_id)
        fs.put(image, _id=item_id)

    def compress_image( self, item_id, image):
        img_obj = Image.open(io.BytesIO(image))
        size = img_obj.size
        if size[0] > 600 or size[1] > 600:
          img_raw = io.BytesIO()
          img_obj.thumbnail((600,600), Image.ANTIALIAS)
          img_obj.save(img_raw, format=img_obj.format, optimize=True, quality=95)
          self.save_image( item_id, img_raw.getvalue())
        return img_obj

    def classify_image( self, image ):

        # based off of https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client.py
        # Compose a JOSN Predict request (send the image tensor).
        # Normalize and batchify the image
        image = image.convert('RGB')
        image = np.expand_dims(np.array(image) / 255.0, 0).tolist()
        predict_request = json.dumps({'instances': image})
        response = requests.post(self.server_url + '/v1/models/resnet:predict', data=predict_request)
        response.raise_for_status()
        return response.json()['predictions'][0]

    def convert_prediction( self, prediction ):
        answers={}
        with open("imagenet_class_index.json", "r") as answer_f:
            answers=json.load(answer_f)

        maxValue=0
        maxIndex=-1
        converted = {}
        for k, v in enumerate(prediction):
            converted[ answers[str(k)][1] ]= v

        return converted

    def tag_predictions( self, item_id, prediction ):
        i = 1
        top_predictions=sorted(prediction.items(), key=lambda x: float(x[1]), reverse=True)[:self.top_x]
        metadata={'predictions':dict(top_predictions)}
        for k, v in top_predictions:
            metadata['prediction%d'%i] = k
            metadata['prediction%d_percentage'%i] = v
            i+=1

        db = self.connect_mongodb()
        item = db.fs.files.find_one( {"_id": item_id} )
        if 'metadata' in item:
            current_metadata = item['metadata']
        else:
            current_metadata = {}

        print(metadata)
        db.fs.files.update_one(
                {'_id': item_id},
                {'$set': {'metadata': current_metadata | metadata }}
            )

    def wait_for_message( self ):
        rabbitmq_channel = self.connect_rabbitmq()
        rabbitmq_channel.queue_declare(queue=self.rabbitmq_queue,durable=True,arguments={"x-queue-type": "quorum"})
        rabbitmq_channel.basic_consume(
                queue=self.rabbitmq_queue,
                on_message_callback=self.rabbitmq_callback,
                auto_ack=False
            )

        # Wait for message
        rabbitmq_channel.start_consuming()


    def run( self ):
       self.wait_for_message()

if __name__ == '__main__':
    RHELDemoWorker().run()
