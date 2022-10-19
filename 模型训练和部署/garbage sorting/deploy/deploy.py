import pika
import base64
from io import BytesIO
from PIL import Image
from test import Model

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='imaginer.fun'))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')

model = Model()
file = open('node.txt', 'r')
node_name = file.read()


def on_request(ch, method, props, body):
    rec = str(body.decode(encoding='utf-8'))
    model_type = int(rec[0])
    image = base64.b64decode(rec[1:])
    image = BytesIO(image)
    image = Image.open(image).convert("RGB")
    label, confidence = model.getLabel(image, model_type)
    response = "这个{}%是{}。此结果由{}计算。".format(int(confidence*10000) / 100, label, node_name)
    print("接受到消息, 预测结果为{}".format(label))
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
