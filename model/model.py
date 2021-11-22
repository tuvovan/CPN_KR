import numpy as np
import tensorflow as tf
import model.config as cfg
from tensorflow import keras
from tensorflow.keras import layers, models, applications


def get_backbone_resnet50(input_shape):
    backbone = applications.ResNet50(
        include_top=False, input_shape=input_shape
    )
    c2_output, c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in [
        "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]

    return keras.Model(
        inputs=[backbone.inputs], outputs=[
            c2_output, c3_output, c4_output, c5_output]
    )


def get_backbone_resnet101(input_shape):
    backbone = applications.ResNet101(
        include_top=False, input_shape=input_shape
    )
    c2_output, c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in [
        "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[
            c2_output, c3_output, c4_output, c5_output]
    )


class GlobalNet(models.Model):
    def __init__(self, backbone=None, **kwargs):
        super().__init__(name='GlobalNet', **kwargs)
        self.backbone = backbone if backbone else get_backbone_resnet50()
        self.conv_c2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c6_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c7_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c8_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c9_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c2_3x3 = keras.layers.Conv2D(
            cfg.NR_SKELETON, 3, 3, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(
            cfg.NR_SKELETON, 3, 3, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(
            cfg.NR_SKELETON, 3, 3, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(
            cfg.NR_SKELETON, 3, 3, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)
        # self.reshape = keras.layers.Resizing(
        #     height=cfg.output_shape[0], width=cfg.output_shape[1])

    def call(self, images, training=False):
        c2_output, c3_output, c4_output, c5_output = self.backbone(
            images, training=training)
        p2_feature = self.conv_c2_1x1(c2_output)
        p3_feature = self.conv_c3_1x1(c3_output)
        p4_feature = self.conv_c4_1x1(c4_output)
        p5_feature = self.conv_c5_1x1(c5_output)

        p4_feature = p4_feature + self.upsample_2x(p5_feature)
        p3_feature = p3_feature + self.upsample_2x(p4_feature)
        p2_feature = p2_feature + self.upsample_2x(p3_feature)

        p2_feature = self.conv_c6_1x1(p2_feature)
        p3_feature = self.conv_c7_1x1(p3_feature)
        p4_feature = self.conv_c8_1x1(p4_feature)
        p5_feature = self.conv_c9_1x1(p5_feature)

        p2_output = self.conv_c2_3x3(p2_feature)
        p3_output = self.conv_c3_3x3(p3_feature)
        p4_output = self.conv_c4_3x3(p4_feature)
        p5_output = self.conv_c5_3x3(p5_feature)

        # p2_output = self.reshape(p2_output)
        # p3_output = self.reshape(p3_output)
        # p4_output = self.reshape(p4_output)
        # p5_output = self.reshape(p5_output)
        p2_output = tf.image.resize(p2_output, size=(cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1]))
        p3_output = tf.image.resize(p3_output, size=(cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1]))
        p4_output = tf.image.resize(p4_output, size=(cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1]))
        p5_output = tf.image.resize(p5_output, size=(cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1]))

        global_feature = [p2_feature, p3_feature, p4_feature, p5_feature]
        global_output = [p2_output, p3_output, p4_output, p5_output]
        return global_feature, global_output

    def build(self, input_shape):
        return super().build(input_shape)


class BottleNeck(layers.Layer):
    def __init__(self, depth, bottleneck_depth, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name,
                         dtype=dtype, dynamic=dynamic, **kwargs)
        self.conv1_11 = keras.layers.Conv2D(
            bottleneck_depth, 1, 1, 'same')
        self.conv1_33 = keras.layers.Conv2D(
            bottleneck_depth, 3, 1, 'same')
        self.conv2_11 = keras.layers.Conv2D(depth, 1, 1, 'same')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_11(inputs)
        x = self.conv1_33(x)
        x = self.conv2_11(x)
        return x + inputs


class RefineNet(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottle_neck_31 = BottleNeck(256, 128)
        self.bottle_neck_41 = BottleNeck(256, 128)
        self.bottle_neck_42 = BottleNeck(256, 128)
        self.bottle_neck_51 = BottleNeck(256, 128)
        self.bottle_neck_52 = BottleNeck(256, 128)
        self.bottle_neck_53 = BottleNeck(256, 128)

        self.bottle_neck_last = BottleNeck(1024, 128)
        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.upsample_4x = keras.layers.UpSampling2D(4)
        self.upsample_8x = keras.layers.UpSampling2D(8)

        self.conv_out = keras.layers.Conv2D(
            cfg.NR_SKELETON, 3, 1, 'same', activation=None)

    def call(self, inputs, training=True, mask=None):
        p2_feature, p3_feature, p4_feature, p5_feature = inputs[0], inputs[1], inputs[2], inputs[3]
        p3_bn = self.bottle_neck_31(p3_feature)
        p4_bn = self.bottle_neck_42(self.bottle_neck_41(p4_feature))
        p5_bn = self.bottle_neck_51(
            self.bottle_neck_52(self.bottle_neck_53(p5_feature)))

        p3_bn = self.upsample_2x(p3_bn)
        p4_bn = self.upsample_4x(p4_bn)
        p5_bn = self.upsample_8x(p5_bn)

        bn_cat = tf.concat([p2_feature, p3_bn, p4_bn, p5_bn], axis=3)
        bn_cat = self.bottle_neck_last(bn_cat)

        final = self.conv_out(bn_cat)
        return final
