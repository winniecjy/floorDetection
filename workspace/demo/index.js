import * as tf from '@tensorflow/tfjs';

const canvas = document.getElementById('data-canvas');
const status = document.getElementById('status');
const testModel = document.getElementById('test');

const BOUNDING_BOX_LINE_WIDTH = 4;
const BOUNDING_BOX_STYLE1 = 'rgb(255,0,0)';
const BOUNDING_BOX_STYLE2 = 'rgb(0,255,0)';


/**
 * 展示预测的楼层位置
 */
function drawBoundingBoxes(canvas, predictBoundingBoxArr) {
  for (const box of predictBoundingBoxArr) {
    let left = box.xmin;
    let right = box.xmax;
    let top = box.ymin;
    let bottom = box.ymax;

    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.strokeStyle = box.label==='floor'?BOUNDING_BOX_STYLE1:BOUNDING_BOX_STYLE2;
    ctx.lineWidth = BOUNDING_BOX_LINE_WIDTH;
    ctx.moveTo(left, top);
    ctx.lineTo(right, top);
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.lineTo(left, top);
    ctx.stroke();

    ctx.font = '15px Arial';
    ctx.fillStyle = box.label==='floor'?BOUNDING_BOX_STYLE1:BOUNDING_BOX_STYLE2;
    ctx.fillText(box.label, left+8, top+8);
  }
}

/**
 * 设计稿图片处理并预测楼层定位
 *
 * @param {tf.Model} model 预测模型
 */
async function runAndVisualizeInference(e, model) {
  if (typeof e === 'string') {
    await new Promise((resolve, reject) => {
      // 图片显示在canvas中
      var img = new Image;
      img.src = e;
      img.onload = function () { //必须onload之后再画
        let w = 500;
        let h = img.height/img.width*500;
        canvas.width = w;
        canvas.height = h;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img,0,0,w,h);
        resolve();
      }
    })
  } else {
    // 上传图片并显示在canvas中
    var file = e.target.files[0]; //获取input输入的图片
    if (!/image\/\w+/.test(file.type)) {
      alert("请确保文件为图像类型");
      return false;
    }
    var reader = new FileReader();
    reader.readAsDataURL(file); //转化成base64数据类型
    await new Promise((resolve, reject) => {
      reader.onload = function (e) {
        // 图片显示在canvas中
        var img = new Image;
        img.src = this.result;
        img.onload = function () { //必须onload之后再画
          let w = 500;
          let h = img.height/img.width*500;
          canvas.width = w;
          canvas.height = h;
          var ctx = canvas.getContext('2d');
          ctx.drawImage(img,0,0,w,h);
          resolve();
        }
      }
    })
  }

  // 模型输入处理
  let image = tf.browser.fromPixels(canvas);
  const t4d = image.expandDims(0);
  /**
   * 所需的输出维度
   * num_detections: 检测总数
   * detection_boxes: 检测框张量
   * detection_scores: 检测框分数，即概率
   * detection_classes: 类别ID，与label_map中相对应
   */
  const outputDim = [
    'num_detections', 'detection_boxes', 'detection_scores',
    'detection_classes'
  ];
  const labelMap = {
    1: 'floor',
    2: 'toutu'
  }
  let modelOut = {}, boxes = [], w = canvas.width, h = canvas.height;
  console.log(model)
  for (const dim of outputDim) {
    let tensor = await model.executeAsync({
      'image_tensor': t4d
    }, `${dim}:0`);
    modelOut[dim] = await tensor.data();
  }
  console.log(modelOut)
  for (let i=0; i<modelOut['detection_scores'].length; i++) {
    const score = modelOut['detection_scores'][i];
    if (score < 0.5) break; // 置信度过滤
    // [ymin , xmin , ymax , xmax]
    boxes.push({
      ymin: modelOut['detection_boxes'][i*4]*h,
      xmin: modelOut['detection_boxes'][i*4+1]*w,
      ymax: modelOut['detection_boxes'][i*4+2]*h,
      xmax: modelOut['detection_boxes'][i*4+3]*w,
      label: labelMap[modelOut['detection_classes'][i]],
    })
  }
  console.log(boxes)

  // 可视化检测框
  drawBoundingBoxes(canvas, boxes);

  // 张量运行内存清除
  tf.dispose([image, modelOut]);
}

async function init() {
  const LOCAL_MODEL_PATH = './web_model/model.json';

  // 将本地模型保存到浏览器
  // tf.sequential().save
  // 加载本地模型
  let model;
  try {
    model = await tf.loadGraphModel(LOCAL_MODEL_PATH);
    // model.summary();
    testModel.disabled = false;
    status.textContent = '成功加载本地模型！请点击"选择文件"按钮上传设计稿';
    runAndVisualizeInference('./test.jpg', model)
  } catch (err) {
    console.log('加载本地模型错误：', err);
    status.textContent = '加载本地模型失败';
  }

  testModel.addEventListener('change', (e) => {
    runAndVisualizeInference(e, model)
  });
}

init();