require('@tensorflow/tfjs-node');

const canvas = require('canvas');
const faceapi = require('@vladmandic/face-api/dist/face-api.node.js');

const {Canvas, Image, ImageData} = canvas;

faceapi.env.monkeyPatch({Canvas, Image, ImageData});
faceapi.nets.ssdMobilenetv1.loadFromDisk('./models/ssd_mobilenetv1');
faceapi.nets.faceLandmark68Net.loadFromDisk('./models/face_landmark_68');
faceapi.nets.faceRecognitionNet.loadFromDisk('./models/face_recognition');

const express = require('express');
const app = express();
const http = require('http').Server(app);
const fileUpload = require('express-fileupload');

app.use(express.json());
app.use(express.urlencoded({extended: true}));
app.use(fileUpload());

function toMultipleFiles(files) {
    if (files && typeof files['data'] !== 'undefined') {
        return [files];
    }

    return files;
}

app.post('/identify', async(request, response) => {
    const token = process.env.ACCESS_TOKEN;

    if (!token || token !== request.query['access_token']) {
        response.status(403).end();

        return;
    }

    if (!request.files) {
        response.status(422).end();

        return;
    }

    const result = [];
    const images = toMultipleFiles(request.files['images']);
    const descriptors = request.body.descriptors;

    for (const index in images) {
        const image = images[index]['data'];
        const descriptor = descriptors ? descriptors[index] : request.body.descriptor;

        result[index] = false;

        if (image && descriptor) {
            try {
                const data = await faceapi.detectSingleFace(await canvas.loadImage(image)).withFaceLandmarks().withFaceDescriptor();
                const matcher = new faceapi.FaceMatcher(data);
                const matching = matcher.findBestMatch(typeof descriptor === 'string' ? JSON.parse(descriptor) : descriptor);

                result[index] = matching.label !== 'unknown';
            } catch (error) {}
        }
    }

    response.json({result});
});

app.post('/descriptors', async(request, response) => {
    const token = process.env.ACCESS_TOKEN;

    if (!token || token !== request.query['access_token']) {
        response.status(403).end();

        return;
    }

    if (!request.files) {
        response.status(422).end();

        return;
    }

    const result = [];
    const images = toMultipleFiles(request.files['images']);

    for (const index in images) {
        const image = images[index]['data'];

        result[index] = null;

        if (image) {
            try {
                const data = await faceapi.detectSingleFace(await canvas.loadImage(image)).withFaceLandmarks().withFaceDescriptor();

                if (data) {
                    result[index] = data.descriptor;
                }
            } catch (error) {}
        }
    }

    response.json({result});
});

http.listen(3000);