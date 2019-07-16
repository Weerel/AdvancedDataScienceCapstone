import numpy as np
from flask import Flask
from flask_restplus import reqparse
from flask_restplus import Api, Resource, fields
from keras.models import load_model
from tensorflow import get_default_graph

app = Flask(__name__)

app.config['RESTPLUS_MASK_SWAGGER'] = False
api = Api(app, version='1.0', title='Advanced Data Science Capstone API',
description='Provides predictions for human measurements',
)

ns = api.namespace('predict', description='Make predictions')

person = api.model('Person', {
    'sex': fields.String(enum=['Male', 'Female'], description='Predicted sex')
})

measurement = reqparse.RequestParser()
measurement.add_argument('age', type=float, help='Age, years')
measurement.add_argument('height', type=float, help='Height, cm')
measurement.add_argument('weight', type=float, help='Weight, kg')
measurement.add_argument('breast', type=float, help='Breast, cm')
measurement.add_argument('waist', type=float, help='Waist, cm')
measurement.add_argument('hips', type=float, help='Hips, cm')

graph = get_default_graph()
sex_classifier = load_model('sex_classifier.h5')

@ns.route('/sex/')
class SexPredictor(Resource):
    @ns.expect(measurement)
    @ns.marshal_with(person)
    def get(self):
        '''Make sex prediction'''
        args = measurement.parse_args()
        
        age = float(args['age'])
        height = float(args['height'])
        weight = float(args['weight'])
        breast = float(args['breast'])
        waist = float(args['waist'])
        hips = float(args['hips'])

        features = np.array([[age, height, weight, breast, waist, hips]])
        with graph.as_default():
            prediction = sex_classifier.predict(features)
        value = prediction[0][0].round().astype(int)
        
        if value==0:
            sex = 'Female'
        else:
            sex = 'Male'

        return {'sex': sex}

if __name__ == "__main__":
    app.run()
