import './three.min.js';
import './OrbitControls.js';
import './RoomEnvironment.js';
import './GLTFLoader.js';
import './RectAreaLightUniformsLib.js';

const visualizer = document.getElementById("visualizer");
const container = document.getElementById('container');
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const pmremGenerator = new THREE.PMREMGenerator(renderer);

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.environment = pmremGenerator.fromScene(new THREE.RoomEnvironment(renderer), 0.04).texture;

const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
directionalLight.position.set(0, 0, 2);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 100);
camera.position.set(5, 2, 8);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.update();
controls.enablePan = true;
controls.enableDamping = true;

// Handle window reseize event
window.onresize = function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
};

var lastFilepath = "";
var needUpdate = false;
var time = 10;

// 更新
function frameUpdate() {
    var filepath = visualizer.getAttribute("filepath");
    if (filepath == lastFilepath) {
        if (needUpdate) {
            controls.update();
            renderer.render(scene, camera);
        }
        requestAnimationFrame(frameUpdate);
    } else {
        needUpdate = false;
        scene.clear();
        progressDialog.open = true;
        lastFilepath = filepath;
        time = 10;
        checkLength();
    }
}

// 定时请求
function checkLength() {
    if (time <= 0) {
        time = 10;
        frameUpdate();
    } else {
        time--;
        loadData();
        setTimeout(checkLength, 500);
    }
}

// 启动定时器
checkLength();

// 循环查询数据是否加载
function loadData() {
    // 循环总时间
    const currentUrl = parent.window.location.href;
    const targetUrl = new URL('/get_obj_Base64', currentUrl);
    const xhr = new XMLHttpRequest();
    xhr.open('GET', targetUrl, true);
    xhr.onload = function () {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                scene.clear();
                progressDialog.open = true;
                main(data);
                visualizer.setAttribute("filepath", data.base64.file);
                time = 0;
            } catch (e) {
                console.error('解析数据失败:', e);
            }
        }
    };
    xhr.onerror = function () {
        console.log('请求错误');
    };

    xhr.send();
}

const onProgress = function (xhr) {
    if (xhr.lengthComputable) {
        progressIndicator.value = xhr.loaded / xhr.total * 100;
    }
};

const onError = function (e) {
    console.error(e);
};

async function main(params) {
    if (params?.base64) {
        const loader = new THREE.GLTFLoader();
        loader.load('data:text/plain;base64,' + params.base64.data, function (gltf) {
            scene.add(gltf.scene);
            gltf.scene.scale.set(5, 5, 5);
            gltf.scene.position.set(0, 0, 0);
        }, onProgress, onError);
        needUpdate = true;
    }

    scene.add(directionalLight);
    scene.add(camera);

    progressDialog.close();

    frameUpdate();
}

main();