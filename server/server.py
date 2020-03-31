#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:47:46 2020

@author: erigara
"""
import aiohttp
from aiohttp import web
from io import StringIO
from prediction import prediction_pipeline_async

import logging
import configparser


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
                prediction = await prediction_pipeline_async(partdata, content_type, self.modeldatapath)
                if prediction:
                    succses = True
                    mpwriter.append(StringIO(prediction),
                                    {'CONTENT-TYPE': content_type})
                    logging.info(f'File: {part.filename} was processed succsesfully')
                else:
                    logging.info(f'File {part.filename} is in unsupported format or invalid')
            if succses:
                status_code=200
                resp = web.StreamResponse(status=status_code, headers={"Content-Type": f'multipart/form-data;boundary={boundary}'})
                await resp.prepare(request)
                await mpwriter.write(resp)
            
        return resp
    

def main():
    configpath = './config/server.conf'
    config = configparser.ConfigParser()
    config.read(configpath)
    
    modeldatapath = config['model']['modeldatapath']
    loggfile = config['logging']['loggfile']
    port = int(config['server']['port'])
    route = config['server']['route']
    
    logging.basicConfig(level = logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(loggfile),
                        logging.StreamHandler()
                    ])
    
    async def init():
        server = await Server.create(modeldatapath)
        app = web.Application()
        app.add_routes([web.post(route, server.predict_handler),])
        return app
    web.run_app(init(), port=port)

if __name__ == "__main__":
    main()
