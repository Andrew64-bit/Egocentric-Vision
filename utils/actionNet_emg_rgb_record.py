class ActionNetVideoEmgRgbRecord(object):
    def __init__(self, descr, label, emg_rgb):
        self._description = descr
        self.emg_rgb = emg_rgb
        self._label = label

    @property
    def duration(self):
        return self.emg_rgb[-1]['time_s'] - self.emg_rgb[0]['time_s']

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label
    
    @property
    def size(self):
        return len(self.emg_rgb)
    
    def get_emg_rgb(self,idx):
        dx = self.emg_rgb[idx]['right_emg']
        sx = self.emg_rgb[idx]['left_emg']
        frame = self.emg_rgb[idx]['frame_name']
        return (dx + sx), frame
        
    def __str__(self):
        return f"Description: {self._description}, Label: {self._label}, Duration: {self.duration}, Size: {self.size}"
