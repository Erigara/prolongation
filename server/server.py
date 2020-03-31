#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:47:46 2020

@author: erigara
"""
import aiohttp
from aiohttp import web
from aiohttp.web import Request, StreamResponse
from aiohttp.multipart import MultipartWriter
import asyncio 
from io import StringIO
import logging
from prediction import prediction_pipeline


logging.basicConfig(level = logging.INFO)

routes = web.RouteTableDef()
class Server:
    async def create(modeldatapath):
        self = Server()
        self.modeldatapath = modeldatapath
        return self
    
    async def predict_handler(self, request):
        '''
        Method create multipart/form-data response that contain 
        all correctly proccesed input files.
        If all files invalid or in wrong formats return 415 error code.

        '''
        boundary = '###boundary###'      
        resp = web.Response(status=415, body='All recieved files invalid or in wrong format')
        with aiohttp.MultipartWriter('multipart/form-data', boundary=boundary) as mpwriter:
            succses = False
            async for part in await request.multipart():
                content_type = part.headers[aiohttp.hdrs.CONTENT_TYPE]
                partdata = await part.text()
                prediction = prediction_pipeline(partdata, content_type, self.modeldatapath)
                if prediction:
                    succses = True
                    mpwriter.append(StringIO(prediction),
                                    {'CONTENT-TYPE': content_type})
                    logging.info(f'File: {part.filename} was processed succsesfully')
                else:
                    logging.info(f'File {part.filename} is in unsupported format or invalid')
            if succses:
                status_code=200
                resp = StreamResponse(status=status_code, headers={"Content-Type": f'multipart/form-data;boundary={boundary}'})
                await resp.prepare(request)
                await mpwriter.write(resp)
            
        return resp
    

def main():
    modeldatapath = '../model/model.joblib.pkl'
    async def init():
        server = await Server.create(modeldatapath)
        app = web.Application()
        app.add_routes([web.post('/predict', server.predict_handler),])
        return app
    web.run_app(init(), port=9000)

if __name__ == "__main__":
    main()