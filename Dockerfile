FROM python:3.8
RUN mkdir prolongation
WORKDIR "prolongation"
ADD server/* server/
ADD install.sh ./
ADD config/* config/
ADD model/* model/
RUN mkdir logg
RUN ls -a
RUN ./install.sh
EXPOSE 9000
CMD "server/server.py"
