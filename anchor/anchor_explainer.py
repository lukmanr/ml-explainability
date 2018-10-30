from anchor import anchor_tabular
import asyncio
import numpy as np
import pandas
import random
import requests
import itertools
import json
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

import utils

REGRESSION = 'regression'
CLASSIFICATION =  'classification'


class Explainer:

    def __cmle_predict(self, record):
        name = 'projects/{}/models/{}'.format(self.__gcp_project, self.__gcp_model)
        if self.__gcp_model_version is not None:
            name += '/versions/{}'.format(self.__gcp_model_version)

        url = 'https://ml.googleapis.com/v1/' + name + ':predict'
        
        result  = requests.post(
            url,
            json={'instances': record},
            headers={
                'Authorization':'Bearer ' + self.__access_token})

        response  = json.loads(result.text)
        try:
            return response['predictions']
        except:
            print('record', record)
            print('response',response)
            return response['predictions']


    def __get_label_values (self):
        cat_labels = utils.load_csv_dataset(
            data=StringIO(self.__csv_file),
            feature_names = self.__feature_names,
            target_idx = -1,
            skip_first = self.__skip_first,
            categorical_features = [],
            features_to_use= [self.__target_idx],
            discretize = True
          ).categorical_names[0]
        self.__categorical_labels = [
            bytes(
                x,
                'utf-8') for x in cat_labels]
        return [float(
            cat_labels[i].split(
                ' <= ')[-1]) for i in range(9)]



    def __create_transform_func(self, quant_val):

        def transfrom_labels(r):
            labels = []
            for x in r:
                x = float(x)
                if x <= quant_val[0]:
                    labels.append(
                        self.__categorical_labels[0])
                if x > quant_val[0] and x <= quant_val[1]:
                    labels.append(
                        self.__categorical_labels[1])
                if x > quant_val[1] and x <= quant_val[2]:
                    labels.append(
                        self.__categorical_labels[2])
                if x > quant_val[2] and x <= quant_val[3]:
                    labels.append(
                        self.__categorical_labels[3])
                if x > quant_val[3] and x <= quant_val[4]:
                    labels.append(
                        self.__categorical_labels[4])
                if x > quant_val[4] and x <= quant_val[5]:
                    labels.append(
                        self.__categorical_labels[5])
                if x > quant_val[5] and x <= quant_val[6]:
                    labels.append(
                        self.__categorical_labels[6])
                if x > quant_val[6] and x <= quant_val[7]:
                    labels.append(
                        self.__categorical_labels[7])
                if x > quant_val[7] and x <= quant_val[8]:
                    labels.append(
                        self.__categorical_labels[8])
                if x > quant_val[8]:
                    labels.append(
                        self.__categorical_labels[9])
            return labels

        return transfrom_labels


    def __get_value_mapper(self):
        self.__col_buckets=[]
        for i in range(0,len(
            self.__explainer.categorical_names)):
            
            if i in self.__dataset.ordinal_features:
            
                cols = [(
                    float('-inf'),
                    float(
                        self.__explainer.categorical_names[i][0].split(' <= ')[-1]))]
                for mid in self.__explainer.categorical_names[i][1:-1]:
                    cols.append((
                        float(mid.split(' < ')[0]),
                        float(mid.split(' <= ')[-1])))
                cols.append((
                    float(
                        self.__explainer.categorical_names[i][-1].split(' > ')[-1]),
                    float('inf')))
                self.__col_buckets.append(cols)
            
            else: 
                self.__col_buckets.append(self.__explainer.categorical_names[i])

        tr_data = pandas.read_csv(
            StringIO(self.__csv_file),
            header = 0,
            names = self.__feature_names,
            delimiter=',',
            na_filter=True,
            dtype=str).fillna("-1").values
        del self.__csv_file
        tr_data = tr_data[:, self.__features_to_use]
       

        val_buckets = []
        for col in range(tr_data.shape[1]):
            
            if col in self.__dataset.ordinal_features:
            
                val_list = {i:[] for i in range(len(self.__col_buckets[col]))}
                for val in tr_data[:,col]:
                    for buck in range(len(self.__col_buckets[col])):
                        if float(
                            val) > self.__col_buckets[col][buck][0] and float(
                            val) <= self.__col_buckets[col][buck][1]:

                            val_list[buck].append(float(val))
                            break
                val_buckets.append(
                [int(np.median(val_list[buck])) if np.median(val_list[buck]).is_integer() else np.median(
                    val_list[buck]) for buck in range(len(self.__col_buckets[col]))])
            else:
                val_buckets.append(self.__col_buckets[col])

        return list(itertools.chain.from_iterable(val_buckets))



    def __encode_record(self, record):
        encoded = []
        for val in range(
            len(record)):
            
            if val in self.__dataset.ordinal_features:
                for buck in range(
                    len(self.__col_buckets[val])):
                    if float(
                        record[val]) > self.__col_buckets[val][buck][0] and float(
                        record[val]) <= self.__col_buckets[val][buck][1]:

                        encoded.append(buck)
                        break
            else:
                for buck in range(
                    len(self.__col_buckets[val])):
                    if record[val] == self.__col_buckets[val][buck]:
                        encoded.append(buck)
                        break
        return np.array(encoded)
            
            
    def __decode_record(self, record):
        return list(itertools.compress(
            self.__value_mapper,
            record.todense().astype(
                int).tolist()[0]))
    
    
    def __predict(self, record):

        if self.__csv_record:
            pred_data = [','.join(self.__pre_pad + [
                str(
                    x) for x in self.__decode_record(
                        record[i,:])] + self.__post_pad) for i in range(
                            record.shape[0])]
        
        else:
            pred_data = [dict(zip(
                self.__dataset.feature_names,
                self.__decode_record(
                record[i,:])))for i in range(
                    record.shape[0])]
        
        predictions = self.__cmle_predict(pred_data)
        predictions = [self.__output_func(x) for x in predictions]
        if self.model_type == REGRESSION:
            predictions = self.__transform_labels(predictions)
        predictions = [self.__label_map[x] for x in predictions]
        predictions = np.array(predictions)
        return predictions

    
    
    def load_data(
        self,
        gcs_path,
        target_idx,
        features_to_use=None,
        categorical_features=[],
        feature_names=None,
        skip_first=False
        ):
        
        self.__target_idx = target_idx
        self.__features_to_use = features_to_use
        self.__feature_names = feature_names
        self.__skip_first = skip_first
        
        self.__numeric_features = list(set(
            features_to_use).difference(
            set(categorical_features)))
        
        self.__csv_file = file_io.FileIO(
            gcs_path,
            mode='r').read()
        
        if self.model_type == REGRESSION:
            self.__transform_labels = self.__create_transform_func(
                self.__get_label_values())
            
            self.__dataset  = utils.load_csv_dataset(
                data=StringIO(self.__csv_file),
                feature_names = feature_names,
                skip_first = skip_first,
                target_idx =target_idx,
                categorical_features = categorical_features,
                features_to_use= features_to_use,
                discretize = True,
                feature_transformations = {
                target_idx : self.__transform_labels
                })
        else:
            self.__dataset  = utils.load_csv_dataset(
                data=StringIO(self.__csv_file),
                feature_names = feature_names,
                skip_first = skip_first,
                target_idx =target_idx,
                categorical_features = categorical_features,
                features_to_use= features_to_use,
                discretize = True)

        
        self.__label_map = {
            self.__dataset.class_names[i] : i for i in range(
                len(self.__dataset.class_names))}
        
        self.__explainer = anchor_tabular.AnchorTabularExplainer(
            self.__dataset.class_names,
            self.__dataset.feature_names,
            self.__dataset.data,
            self.__dataset.categorical_names,
            self.__dataset.ordinal_features)
        
        self.__explainer.fit(
            self.__dataset.train,
            self.__dataset.labels_train,
            self.__dataset.validation,
            self.__dataset.labels_validation)
        
        self.__value_mapper = self.__get_value_mapper()
        
    def create_cmle_client(
        self,
        gcp_project,
        gcp_model,
        access_token,
        gcp_model_version=None,
        csv_record=True,
        padding = (1,0),
        output_func = lambda x: x[
            'predictions'][0]):
    
        self.__gcp_project = gcp_project
        self.__gcp_model = gcp_model
        self.__csv_record = csv_record
        self.__gcp_model_version = gcp_model_version
        self.__pre_pad = ['0' for _ in range(padding[0])]
        self.__post_pad = ['0' for _ in range(padding[1])]
        self.__output_func = output_func
        self.__access_token = access_token
    
    
    def __assess_sample(
        self,
        idx):
        if self.__explainer.class_names[
                self.__dataset.labels_test[idx]] == self.__predict_record(
                self.__one_hot_encode(
                    self.__dataset.test[idx])):
            return 1
        else:
            return 0
        
    
    async def __assess_model(
        self,
        sample):
        
        loop = asyncio.get_event_loop()
        samples = [
            loop.run_in_executor(
                None,
                self.__assess_sample,
                idx) for idx in random.sample(
                range(len(self.__dataset.test)),
                sample)]
        responses = [await s for s in samples]
        
        return sum(responses)
    
    def assess_model(
        self,
        sample=100):
        
        self.__check_requisites()
        
        accurate = self.__event_loop.run_until_complete(
            self.__assess_model(sample))

        return {'accuracy' : accurate/float(sample)}

    
    def __explain_sample(
        self,
        idx,
        threshold):

        anchor = {}

        anchor['prediction'] = self.__predict_record(
            self.__one_hot_encode(
                self.__dataset.test[idx]))
        exp = self.__explainer.explain_instance(
            self.__dataset.test[idx],
            self.__predict,
            threshold,
            delta=0.05, tau=0.1, batch_size=10,
            max_anchor_size=10)
            
        anchor['precision'] = exp.precision()
        anchor['coverage'] = exp.coverage()
        anchor['anchor'] = ' AND '.join(exp.names())

        return anchor
    
    
    async def __explain_model(
        self,
        threshold,
        sample):

        loop = asyncio.get_event_loop()
        samples = [
            loop.run_in_executor(
                None,
                self.__explain_sample,
                idx,
                threshold) for idx in random.sample(
                range(len(self.__dataset.test)),
                sample)]
        return [await s for s in samples]

    
    def explain_model(
        self,
        threshold = 0.95,
        sample = 100):
        
        self.__check_requisites()
        
        anchors = self.__event_loop.run_until_complete(
            self.__explain_model(threshold, sample))
                    
        return pandas.DataFrame(
            anchors).drop_duplicates(
            ['anchor']).sort_values(
            'coverage',
            ascending = False).reset_index()[[
            'anchor',
            'coverage',
            'precision',
            'prediction']]
            
            
    def __explain_record(
        self,
        record,
        threshold,
        delta,
        tau,
        batch_size,
        max_anchor_size,
        show_in_notebook = False):
        
        one_hot_record = self.__one_hot_encode(record)

        pred = self.__predict_record(one_hot_record)

        exp = self.__explainer.explain_instance(
            record,
            self.__predict,
            threshold,
            delta=delta,
            tau=tau,
            batch_size=batch_size,
            max_anchor_size=max_anchor_size)

        if show_in_notebook:
            exp.show_in_notebook()

        return {
            'anchor': ' AND '.join(exp.names()),
            'precision': exp.precision(),
            'coverage' : exp.coverage(),
            'prediction' : pred,
            'record': self.__decode_record(one_hot_record)}

    
    def explain_record(
        self,
        record,
        threshold = 0.95,
        delta=0.05,
        tau=0.2,
        batch_size=100,
        max_anchor_size=20,
        show_in_notebook = False):
        
        self.__check_requisites()
        
        record = self.__encode_record(
            record)
        
        return self.__explain_record(
            record,
            threshold,
            delta,
            tau,
            batch_size,
            max_anchor_size,
            show_in_notebook)
        
        
    def explain_random_record(
        self,
        threshold = 0.95,
        delta=0.05,
        tau=0.2,
        batch_size=100,
        max_anchor_size=20,
        show_in_notebook = False):
        
        self.__check_requisites()
        
        idx = random.randint(0,len(self.__dataset.test))
        return self.__explain_record(
            self.__dataset.test[idx],
            threshold,
            delta,
            tau,
            batch_size,
            max_anchor_size,
            show_in_notebook)
        
                   
    def __check_requisites(self):
        if not '_Explainer__explainer' in self.__dict__:
            raise Exception(
                'Please load a dataset using load_data(...)')
        if not '_Explainer__gcp_model' in self.__dict__:
            raise Exception(
                'Please create a cmle client using create_cmle_client(...)')
    
    def __one_hot_encode(self, record):

        return self.__explainer.encoder.transform(
                    record.reshape(1, -1))
    
    def __predict_record(
        self,
        record):
        return self.__explainer.class_names[list(
            self.__predict(record))[0]]
        

    def __init__(
        self,
        model_type = REGRESSION
    ):
        self.model_type = model_type
        self.__event_loop =  asyncio.new_event_loop()
        
