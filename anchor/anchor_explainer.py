from anchor import utils
from anchor import anchor_tabular
import numpy as np
import pandas
import random
import itertools
import googleapiclient.discovery
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO


class Explainer:

    def __cmle_predict(self, record):
        name = 'projects/{}/models/{}'.format(self.__gcp_project, self.__gcp_model)
        if self.__gcp_model_version is not None:
            name += '/versions/{}'.format(self.__gcp_model_version)

        response = self.__cmle_service.predict(
            name=name,
            body={'instances': record}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']


    def __get_label_values (self):
        self.__categorical_labels = utils.load_csv_dataset(
            data=StringIO(self.__csv_file),
            feature_names = self.__feature_names,
            target_idx = -1,
            skip_first = self.__skip_first,
            categorical_features = [],
            features_to_use= [self.__target_idx],
            discretize = True
          ).categorical_names[0]
        return [float(
            self.__categorical_labels[i].split(
                ' <= ')[-1]) for i in range(3)]



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
                if x > quant_val[2]:
                    labels.append(
                        self.__categorical_labels[3])
            return labels

        return transfrom_labels


    def __get_value_mapper(self):
        col_buckets=[]
        for i in range(0,len(
            self.__explainer.categorical_names)):
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
            col_buckets.append(cols)

        tr_data = pandas.read_csv(
            StringIO(self.__csv_file),
            header= None if self.__skip_first else 0,
            names = self.__feature_names,
            delimiter=',',
            na_filter=True,
            dtype=str).fillna("-1").values
        del self.__csv_file
        tr_data = tr_data[:, self.__features_to_use]
       

        val_buckets = []
        for col in range(tr_data.shape[1]):
            val_list = {i:[] for i in range(len(col_buckets[col]))}
            for val in tr_data[:,col]:
                for buck in range(len(col_buckets[col])):
                    if float(val) > col_buckets[col][buck][0] and float(val) <= col_buckets[col][buck][1]:
                        val_list[buck].append(float(val))
                        break
            val_buckets.append(
              [str(int(np.median(val_list[buck]))) if np.median(val_list[buck]).is_integer() else str(
                  np.median(val_list[buck])) for buck in range(len(col_buckets[col]))])
       
        return list(itertools.chain.from_iterable(val_buckets))


    def __predict(self, record):
        pred_data = [
            ','.join(['1']+ list(
                itertools.compress(
                    self.__value_mapper,
                    record.todense()[i,:].astype(int).tolist()[0]))) for i in range(record.shape[0])]
        predictions = self.__cmle_predict(pred_data)
        predictions = [x['predicted_monetary'] for x in predictions]
        predictions = self.__transform_labels(predictions)
        predictions = [self.__label_map[x] for x in predictions]
        predictions = np.array(predictions)
        return predictions

    
    
    def load_data(
        self,
        gcs_path,
        target_idx,
        features_to_use=None,
        feature_names=None,
        skip_first=False
        ):
        
        self.__target_idx = target_idx
        self.__features_to_use = features_to_use
        self.__feature_names = feature_names
        self.__skip_first = skip_first
        
        self.__csv_file = file_io.FileIO(
            gcs_path,
            mode='r').read()
        
        self.__transform_labels = self.__create_transform_func(
            self.__get_label_values())
        
        self.__dataset  = utils.load_csv_dataset(
            data=StringIO(self.__csv_file),
            feature_names = feature_names,
            skip_first = skip_first,
            target_idx =target_idx,
            categorical_features = [],
            features_to_use= features_to_use,
            discretize = True,
            feature_transformations = {
              target_idx : self.__transform_labels
            }
          )
        
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
        gcp_model_version=None):
        self.__gcp_project = gcp_project
        self.__gcp_model = gcp_model
        self.__gcp_model_version = gcp_model_version
        self.__cmle_service = googleapiclient.discovery.build(
            'ml', 'v1').projects()
    
    
    def assess_model(
        self,
        sample=100):
        
        self.__check_requisites()
        
        accurate = 0
        for idx in random.sample(
            range(len(self.__dataset.test)),
            sample):
            
            if self.__explainer.class_names[
                self.__dataset.labels_test[idx]] == self.__predict_record(
                self.__dataset.test[idx]):
                
                accurate = accurate + 1
        
        return {'accuracy' : accurate/float(sample)}
        
    
    def __predict_random(self):
        
        idx = random.randint(0,len(self.__dataset.test))
        
        
        prediction = {'prediction': self.__predict_record(
            self.__dataset.test[idx])}
        
 
        prediction['label'] = self.__explainer.class_names[
            self.__dataset.labels_test[idx]]
    
        prediction['idx'] = idx
        
        return prediction

    
    def explain_model(
        self,
        threshold = 0.95,
        sample = 100):
        
        self.__check_requisites()
        
        anchors = {
            'anchor': [],
            'precision': [],
            'coverage' : [],
            'prediction' : []}
        
        for idx in random.sample(
            range(len(self.__dataset.test)),
            sample):
            
            
            anchors['prediction'].append(
                self.__predict_record(
                    self.__dataset.test[idx]))
            
            exp = self.__explainer.explain_instance(
            self.__dataset.test[idx],
            self.__predict,
            threshold)
            
            anchors['precision'].append(exp.precision()) 
            anchors['coverage'].append(exp.coverage()) 
            anchors['anchor'].append(
                ' AND '.join(exp.names())) 
                    
            
        return pandas.DataFrame(
            anchors).drop_duplicates(
            ['anchor']).sort_values(
            'coverage',
            ascending = False).reset_index()[[
            'anchor',
            'coverage',
            'precision',
            'prediction']]
            
            
    def explain_random_record(
        self,
        threshold = 0.95,
        show_in_notebook = False):
        
        self.__check_requisites()
        
        random_pred = self.__predict_random()
        
        random_pred['exp'] = self.__explainer.explain_instance(
            self.__dataset.test[random_pred['idx']],
            self.__predict,
            threshold)
        
        if show_in_notebook:
            random_pred['exp'].show_in_notebook()
            
        return {
            'anchor': ' AND '.join(random_pred['exp'].names()),
            'precision': random_pred['exp'].precision(),
            'coverage' : random_pred['exp'].coverage(),
            'prediction' : random_pred['prediction']}
        
                   
    def __check_requisites(self):
        if not '_Explainer__explainer' in self.__dict__:
            raise Exception(
                'Please load a dataset using load_data(...)')
        if not '_Explainer__cmle_service' in self.__dict__:
            raise Exception(
                'Please create a cmle client using create_cmle_client(...)')
    
    def __predict_record(
        self,
        record):

        return self.__explainer.class_names[list(
            self.__predict(
                self.__explainer.encoder.transform(
                    record.reshape(1, -1))))[0]]
        

    def __init__(
        self,
        model_type = 'regression'
    ):
        self.model_type = model_type
        
