import lime
import random
import sklearn
import numpy as np
import pandas as pd
import lime.lime_tabular
import googleapiclient.discovery

from tensorflow.python.lib.io import file_io



class Explainer:

    def __init__(self, model_type='regression'):
        self.model_type = model_type

    def load_data(
            self,
            gcs_path,
            features_to_use=None,
            categorical_features=[],
            feature_names=None,
            skip_rows=0
    ):

        # self.__target_idx = target_idx
        self.__features_to_use = features_to_use
        self.__skip_rows = skip_rows

        with file_io.FileIO(gcs_path, 'r') as f:
            data = pd.read_csv(f,
                               header=None,
                               delimiter=",",
                               na_filter=True,
                               dtype=str,
                               skiprows=self.__skip_rows)

        self.__data = data.iloc[:100]
        data = data.iloc[:, self.__features_to_use]
        self.__categorical_features = [self.__features_to_use.index(i)
                                       for i in categorical_features]
        self.__feature_names = [feature_names[i] for i in
                                self.__features_to_use]
        data, categorical_names, le_list = self.__encode_categorical(
            data)

        self.__categorical_names = categorical_names
        self.__le_list = le_list
        self.__dataset = data.values.astype(float)

        self.__explainer = lime.lime_tabular.LimeTabularExplainer(
            self.__dataset,
            feature_names=self.__feature_names,
            categorical_features=self.__categorical_features,
            categorical_names=self.__categorical_names,
            verbose=True,
            mode=self.model_type,
        )

    def __encode_categorical(self, data):
        categorical_names = {}
        le_list = []
        for feature in self.__categorical_features:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(data.iloc[:, feature])
            data.iloc[:, feature] = le.transform(data.iloc[:, feature])
            categorical_names[feature] = le.classes_
            le_list.append(le)
        return data, categorical_names, le_list

    def create_cmle_client(
            self,
            gcp_project,
            gcp_model,
            gcp_model_version=None,
            padding=(1, 0),
            output_func=lambda x: x[
                'predictions'][0]):

        self.__gcp_project = gcp_project
        self.__gcp_model = gcp_model
        self.__gcp_model_version = gcp_model_version
        self.__pre_pad = ['0' for _ in range(padding[0])]
        self.__post_pad = ['0' for _ in range(padding[1])]
        self.__output_func = output_func
        self.__cmle_service = googleapiclient.discovery.build(
            'ml', 'v1').projects()

    def __cmle_predict(self, record):
        name = 'projects/{}/models/{}'.format(self.__gcp_project,
                                              self.__gcp_model)
        if self.__gcp_model_version is not None:
            name += '/versions/{}'.format(self.__gcp_model_version)

        response = self.__cmle_service.predict(
            name=name,
            body={'instances': record}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']

    def __decode_record(self, row):
        row = row.astype(str)
        for feature, le in zip(self.__categorical_features,
                               self.__le_list):
            row[feature] = \
            le.inverse_transform([int(float(row[feature]))])[0]
        for idx in self.__numeric_rows:
            row[idx] = int(float(row[idx]))
        return row.tolist()

    def __pred_fn(self, record):
        pred_data = [','.join(self.__pre_pad + self.__decode_record(
            record[i, :]) + self.__post_pad) for i in range(
            record.shape[0])]
        predictions = self.__cmle_predict(pred_data)
        predictions = [self.__output_func(x) for x in predictions]
        predictions = np.array(predictions)
        return predictions

    def __explain_record(
            self,
            record,
            num_features=5,
            num_samples=10,
            show_in_notebook=False,
            need_to_encode=False
    ):

        record = self.__encode_record(record, need_to_encode)
        exp = self.__explainer.explain_instance(record, self.__pred_fn,
                                                num_features=num_features,
                                                num_samples=num_samples)

        if show_in_notebook:
            exp.show_in_notebook()

        list_of_vals = exp.as_list()
        df = pd.DataFrame({
            'representation': [rep for rep, _ in list_of_vals],
            'weight': [weight for _, weight in list_of_vals]})
        return record.tolist(), df

    def __encode_record(self, record, need_to_encode):
        record = np.array(record.split(","))
        if need_to_encode:
            record = record[self.__features_to_use]
            for feature, le in zip(self.__categorical_features,
                                   self.__le_list):
                record[feature] = \
                    le.transform(
                        np.array(record[feature]).reshape(-1, ))[0]
        return record.astype(float)

    def explain_record(
            self,
            record,
            show_in_notebook=False,
            num_features=5,
            num_samples=10):

        self.__check_requisites()

        return self.__explain_record(
            record,
            num_features,
            num_samples,
            show_in_notebook)

    def explain_random_record(
            self,
            numeric_rows,
            show_in_notebook=False,
            num_features=5,
            num_samples=10
    ):

        self.__check_requisites()
        self.__numeric_rows = numeric_rows
        idx = random.randint(0, len(self.__data))

        return self.__explain_record(
            ','.join(str(e) for e in list(self.__data.iloc[idx, :])),
            num_features,
            num_samples,
            show_in_notebook,
            need_to_encode=True)

    def __check_requisites(self):
        if not '_Explainer__explainer' in self.__dict__:
            raise Exception(
                'Please load a dataset using load_data(...)')
        if not '_Explainer__cmle_service' in self.__dict__:
            raise Exception(
                'Please create a cmle client using create_cmle_client(...)')
