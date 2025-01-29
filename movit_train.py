import os
import argparse
import tensorflow as tf
import random
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import tensorflow_hub as hub
import tensorflow_models as tfm
import matplotlib.pyplot as plt

class FrameGenerator:
    def __init__(self, dataset_path: str, resolution:int):
        self.dataset_path = dataset_path
        self.segments = os.listdir(dataset_path)
        random.shuffle(self.segments)
        self.resolution = resolution

    def __call__(self):
        random_rotation = random.randint(-180, 180)
        random_crop = random.uniform(0.7,1)
        flip_h:bool = random.random()>0.5
        flip_v:bool = random.random()>0.5
        brightness = random.uniform(-0.2,0.2)
        saturation = random.uniform(0,1.5)
        hue = random.uniform(-0.2,0.2)
        brightness = random.uniform(-0.2,0.2)
        contrast = random.uniform(0.8,3)

        def apply_aug(img):
            img=tfm.vision.augment.rotate(img, random_rotation)
            img = tf.image.central_crop(img, random_crop)
            if flip_h:
                img = tf.image.flip_left_right(img)
            if flip_v:
                img = tf.image.flip_up_down(img)
            img = tf.image.adjust_brightness(img, delta=brightness)
            img = tf.image.adjust_saturation(img,saturation)
            img = tf.image.adjust_hue(img,hue)
            img = tf.image.adjust_contrast(img,contrast)
            return img
        
        for segment in self.segments:
            segment_abs = os.path.join(self.dataset_path, segment)
            image_paths = sorted([os.path.join(segment_abs, img) for img in os.listdir(segment_abs)])
            frames = [tf.image.decode_png(tf.io.read_file(img), channels=3) for img in image_paths]
            frames = [tf.image.resize(frame, (self.resolution, self.resolution)) for frame in frames]  # Ridimensiona le immagini, se necessario
            frames = [apply_aug(frame) for frame in frames]
            yield tf.stack(frames), 1. if "infested" in segment else 0.


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--devices", type=str, default="0,1,2,3")
parser.add_argument("--window_size",type=int, default=32)
parser.add_argument("--model_name",type=str, default="a2")
parser.add_argument("--resolution", type=int, default=224)

args = parser.parse_args()

random.seed(1234)

train_fg = FrameGenerator(os.path.join(args.dataset,"train"), resolution=args.resolution)
val_fg = FrameGenerator(os.path.join(args.dataset,"val"), resolution=args.resolution)
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.float32))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_generator(train_fg, output_signature = output_signature)
val_ds = tf.data.Dataset.from_generator(val_fg, output_signature = output_signature)
train_ds = train_ds.prefetch(buffer_size = AUTOTUNE).batch(args.batch_size)
val_ds = val_ds.prefetch(buffer_size = AUTOTUNE).batch(args.batch_size)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

tf.keras.backend.clear_session()

# Create backbone and model.
backbone = movinet.Movinet(
    model_id='a0',
    causal=False,
    use_external_states=False,
)
model = movinet_model.MovinetClassifier(
    backbone, num_classes=1, output_states=False)
model.build([args.batch_size, args.window_size, args.resolution, args.resolution, 3])

loss_obj =  tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
model.compile(loss=loss_obj, optimizer='adam', metrics=['accuracy'])
results = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=args.epochs)