"""
Switch stores the position of the value in the feature map before pooling.
"""


class Switch:
    """
    used to record switch information of pooling layer
    """

    def __init__(self):
        self.switches = {}

    def add_switch(self, layer_index, switch):
        """
        add switch information
        :param layer_index: layer index
        :param switch: switch information of current layer
        """
        self.switches[layer_index] = switch

    def get_switch(self, layer_index):
        """
        get switch information of specified layer
        :param layer_index: layer index
        :return: switch information
        """
        return self.switches.get(layer_index, dict())

    def get_switch_selected_img(self, layer_index, selected_batch):
        return self.switches.get(layer_index, dict())[selected_batch]
