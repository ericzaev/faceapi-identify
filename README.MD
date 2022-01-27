`docker run -d -p 3000:3000 --env ACCESS_TOKEN=123456789 --name=faceapi-identify --restart=always eskater/faceapi-identify:latest`

### used models

```
faceapi.nets.ssdMobilenetv1.loadFromDisk('./models/ssd_mobilenetv1');
faceapi.nets.faceLandmark68Net.loadFromDisk('./models/face_landmark_68');
faceapi.nets.faceRecognitionNet.loadFromDisk('./models/face_recognition');
```

### envs

1. ACCESS_TOKEN=null

### expressjs
1. post `/identify?access_token=123456789` body `{images: [], descriptor: null, descriptors: []}` return `[true, ...]`
2. post `/descriptors?access_token=123456789` body `{images: []}` return `[descriptor, ...]`

### nginx
```
location /faceapi/ {
  proxy_pass http://faceapi-identify:3000/;

  proxy_http_version 1.1;
  proxy_set_header Host $host;
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";
  proxy_set_header X-NginX-Proxy true;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

### docker-compose
```
faceapi:
    image: eskater/faceapi-identify:latest
    container_name: app_faceapi-identify
    environment:
        - "ACCESS_TOKEN=${FACEAPI_ACCESS_TOKEN}"
    networks:
        - backend
```