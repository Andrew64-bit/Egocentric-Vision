from .actionNet_emg_rgb_record import ActionNetVideoEmgRgbRecord
import pandas as pd


class ActionNetVideoEmgRecord(object):
    def __init__(self, descr, label, emg):
        self._description = descr
        self.emg = emg
        self._label = label

    @property
    def duration(self):
        return self.emg[-1]['time_s'] - self.emg[0]['time_s']

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label
    
    @property
    def size(self):
        return len(self.emg)
    
    def get_emg(self,idx):
        dx = self.emg[idx]['right_emg']
        sx = self.emg[idx]['left_emg']
        return dx + sx
    
    def combine_emg_rgb(self, rgb):
        descr = self.description
        label = self.label
        emg = pd.DataFrame(self.emg)
        result_df = pd.merge_asof(emg.sort_values('time_s'), rgb.sort_values('time_s'), on='time_s', direction='nearest')
        list_of_dicts = result_df.to_dict(orient='records')
        return ActionNetVideoEmgRgbRecord(descr, label, list_of_dicts)
    
    def __str__(self):
        return f"Description: {self._description}, Label: {self._label}, Duration: {self.duration}, Size: {self.size}"
