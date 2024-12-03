import torch.nn as nn
from networks.reconstrcutFilter import ReconstructFilter
from networks.rectify import Rectify
from networks.unPooling import Unpooling
import torch


class DeconvNet(nn.Module):
    def __init__(self, conv_model):
        super(DeconvNet, self).__init__()
        self.conv_model = conv_model
        # self.deconv_layers = nn.ModuleList()

        for i, layer in enumerate(self.conv_model.children()):
            # only deconv the first 5 layers
            if i == 6:
                break

            if isinstance(layer, nn.Sequential):
                deconv_seq = nn.Sequential()
                for _, sub_layer in enumerate(reversed(list(layer.children()))):
                    if isinstance(sub_layer, nn.Conv2d):
                        deconv_seq.add_module(
                            f"deconv_{sub_layer.__class__.__name__}_{i}",
                            ReconstructFilter(sub_layer),
                        )
                    elif isinstance(sub_layer, nn.ReLU):
                        deconv_seq.add_module(
                            f"deconv_{sub_layer.__class__.__name__}_{i}", Rectify()
                        )
                    elif isinstance(sub_layer, nn.MaxPool2d):
                        deconv_seq.add_module(
                            f"deconv_{sub_layer.__class__.__name__}_{i}",
                            Unpooling(sub_layer),
                        )
                self.add_module(f"deconv_{i}", deconv_seq)

    def get_feature_map(self, img_batch, layer_index):
        feature_map = img_batch
        max_layer_input_shape = {}
        for i, layer in enumerate(self.conv_model.children()):
            for _, sub_layer in enumerate(layer.children()):
                if isinstance(sub_layer, nn.MaxPool2d):
                    max_layer_input_shape[i] = (
                        feature_map.shape[2],
                        feature_map.shape[3],
                    )

                feature_map = sub_layer(feature_map)
            if i == layer_index:
                break
        return feature_map, max_layer_input_shape

    def select_channels(self, feature_map, top_k=1):
        """
        select top 1 channels with highest average activation value
        """
        batch_size, channels, _, _ = feature_map.shape

        # calculate the importance of each channel
        channel_importance = (
            feature_map.view(batch_size, channels, -1).max(dim=2).values.mean(dim=0)
        )  # importance defined by the average of max activation value of each channel of each batch
        selected_channels = channel_importance.topk(k=top_k).indices

        return selected_channels

    def zero_out_channels(self, feature_map, selected_channel):
        """
        zero out the channels that are not in the selected channel and keep only the maximum activation value in the selected channel
        """
        selected_map = torch.zeros_like(feature_map)
        # get the indices of the maximum activation value in the selected channel
        max_indices = (
            feature_map[:, selected_channel, :, :]
            .view(feature_map.size(0), -1)
            .argmax(dim=1)
        )
        # set the maximum activation value to the selected map
        for batch_idx in range(feature_map.size(0)):
            max_pos = torch.unravel_index(
                max_indices[batch_idx], (feature_map.size(2), feature_map.size(3))
            )
            selected_map[batch_idx, selected_channel, max_pos[0], max_pos[1]] = (
                feature_map[batch_idx, selected_channel, max_pos[0], max_pos[1]]
            )

        return selected_map

    def remove_images(self, feature_map, selected_channel):
        """
        only keep the top 9 images with highest activation value of the selected channel
        """
        batch_size, channels, height, width = feature_map.shape
        # Create a mask to zero out unwanted images
        mask = torch.zeros_like(feature_map)
        selected_batch_idx = []

        # Calculate the maximum activation value for each image in the current channel
        max_activations = (
            feature_map[:, selected_channel, :, :]
            .view(batch_size, -1)
            .max(dim=1)
            .values
        )
        # Get the indices of the top 9 images with the highest activation values
        top_images_indices = max_activations.topk(k=9).indices
        # Set the mask for the top images to 1 for all channels
        mask[top_images_indices, :, :, :] = 1

        # Collect the indices of the selected images
        selected_batch_idx.extend(top_images_indices.tolist())

        # Apply the mask to the feature map
        feature_map = feature_map * mask

        # Remove images not in the selected indices
        unique_selected_batch_idx = list(set(selected_batch_idx))
        feature_map = feature_map[unique_selected_batch_idx]

        return feature_map, selected_batch_idx

    def deconv_layer_by_layer(
        self,
        feature_map,
        max_layer_input_shape,
        switches,
        layer_index,
        selected_batch_idx,
    ):
        # deconv the feature map layer by layer
        for i in range(layer_index, -1, -1):
            deconv_layer = getattr(self, f"deconv_{i}")
            for _, sub_layer in enumerate(deconv_layer.children()):
                if isinstance(sub_layer, Unpooling):
                    switches_i = switches.get_switch_selected_img(i, selected_batch_idx)
                    feature_map = sub_layer(
                        feature_map, switches_i, max_layer_input_shape[i]
                    )
                elif isinstance(sub_layer, Rectify):
                    feature_map = sub_layer(feature_map)
                elif isinstance(sub_layer, ReconstructFilter):
                    feature_map = sub_layer(feature_map)

        return feature_map

    def find_kernal(self, selected_channel):
        # get layer 0 conv layer from conv_model
        conv_layer = self.conv_model.layer0[0]

        # extract kernel weights, every channel has a kernel weight and all images in the batch share the same kernel weight
        kernel_weights = conv_layer.weight.data[selected_channel]

        return kernel_weights

    def forward(self, img_batch, layer_index):
        """
        execute layer by layer deconvolution operation
        :param img: input imagem ndarray, shape [b, c, h, w]
        :param layer_index: start deconvolution from the specified layer, start from 0
        :return: feature mapped to the input space
        """
        # input image must be 224x224, 3 channels
        if (
            img_batch.shape[1] != 3
            or img_batch.shape[2] != 224
            or img_batch.shape[3] != 224
        ):
            raise ValueError("Input image must be of size 224x224, 3 channels")

        # get the feature map output of the specified layer
        feature_map, max_layer_input_shape = self.get_feature_map(
            img_batch, layer_index
        )
        switches = self.conv_model.switches

        selected_channels = self.select_channels(
            feature_map, top_k=9 if layer_index == 0 else 1
        )
        output_feature_maps = []
        output_selected_batch_idx = []
        output_kernals = []
        for selected_channel in selected_channels:
            feature_map_c = feature_map.clone()
            feature_map_c, selected_batch_idx = self.remove_images(
                feature_map_c, selected_channel
            )
            feature_map_c = self.zero_out_channels(feature_map_c, selected_channel)
            kernal = None

            if layer_index == 0:
                kernal = self.find_kernal(selected_channel)
                feature_map_c = self.deconv_layer_by_layer(
                    feature_map_c,
                    max_layer_input_shape,
                    switches,
                    layer_index,
                    selected_batch_idx,
                )
            else:
                feature_map_c = self.deconv_layer_by_layer(
                    feature_map_c,
                    max_layer_input_shape,
                    switches,
                    layer_index,
                    selected_batch_idx,
                )

            output_feature_maps.append(feature_map_c)
            output_selected_batch_idx.append(selected_batch_idx)
            output_kernals.append(kernal)

        output_feature_map = {
            "feature_map": output_feature_maps,
            "selected_batch_idx": output_selected_batch_idx,
            "kernal": output_kernals,
        }
        return output_feature_map
