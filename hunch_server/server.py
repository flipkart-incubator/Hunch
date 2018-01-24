import json
import os
import traceback
from flask import Flask, request, jsonify
from model_loader import ModelLoader
import requests
import hunch_server
import yaml

hunch_server_config = {}

if 'HUNCH_CONFIG' in os.environ:
    if os.path.exists(os.environ['HUNCH_CONFIG']):
        with open(os.environ['HUNCH_CONFIG']) as f:
            hunch_server_config = yaml.load(f)

ROTATION_FILE_PATH = hunch_server_config["rotation_status_file"]
app = Flask(__name__)
app.logger_name = "hunch.app"
models_loaded = {}
model_loader = ModelLoader(hunch_server_config)
try:
    if 'MODELS_TO_LOAD' in os.environ:
        models_to_load = json.loads(os.environ['MODELS_TO_LOAD'])
        models_loaded = model_loader.get_models_from_list(models_to_load)

except requests.exceptions.HTTPError as e:
    app.logger.error("Meta Service has thrown %s, the error is %s and stack trace is %s"
                     %(e.response.status_code, e.message, str(traceback.format_exc())))
    raise RuntimeError("Meta Service has thrown '{}' , the error is {} and stack trace is {}".format(e.response.status_code, e.message, str(traceback.format_exc())))

app.logger.info("Loaded models are: " + json.dumps(models_loaded.keys()))

@app.route('/elb-healthcheck', methods=['GET'])
def elb_healthcheck():
    try:
        if os.path.isfile(ROTATION_FILE_PATH):
            with open(ROTATION_FILE_PATH) as fd:
                lines = (fd.read()).strip()
                if lines == '1':
                    #TODO: uptime, requests and capacity have to be computed
                    result = {"uptime": 0, "requests": 0, "capacity": 100}
                    return jsonify(result)
        response = jsonify({'Message': "Out of rotation"})
        response.status_code = 500
        return response
    except Exception as e:
        response = jsonify({'Message': "Out of rotation"})
        response.status_code = 500
        return response

@app.route('/rotation_status', methods=['POST'])
def rotation_status():
    try:
        state = request.args.get('state')
        if state is not None:
            if state == "oor" or state == "OOR":
                write_rotation_status(ROTATION_FILE_PATH, '0')
                result = {'Message': "Taking out of rotation"}
                return jsonify(result)
            elif state == "bir" or state == "BIR":
                write_rotation_status(ROTATION_FILE_PATH, '1')
                result = {'Message': "Taking back in rotation"}
                return jsonify(result)
            else:
                response = jsonify({'Message': "Bad Request"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'Message': "Bad Request"})
            response.status_code = 400
            return response
    except Exception as e:
        result = {'Message': str(e)}
        return jsonify(result)


@app.route('/models-loaded', methods=['GET'])
def models_available():
    response = jsonify({"result":models_loaded.keys()})
    response.status_code = 200
    return response

@app.route('/health', methods=['GET'])
def health():
    app.logger.debug("Health Check")
    try:
        if os.path.isfile(ROTATION_FILE_PATH):
            with open(ROTATION_FILE_PATH) as fd:
                lines = (fd.read()).strip()
                if lines == '1':
                    result = {"version": hunch_server.__version__, "health_status": "OK"}
                    return json.dumps(result)
        response = jsonify({'Message': "Out of rotation"})
        response.status_code = 500
        return response
    except Exception as e:
        exc_traceback = str(traceback.format_exc())
        app.logger.error("Exception occurred: " + str(e.message) + "," + exc_traceback)
        response = jsonify({"stack_trace": exc_traceback})
        response.status_code = 500
        return response


@app.route('/predict', methods=['POST'])
def predict():
    try:
        try:
            input = json.loads(request.data)
        except ValueError as e:
            stack_trace = traceback.format_exc()
            app.logger.error("Json Decoding failed. Check if the payload is correct. Payload is " +
                             str(request.data) + " and the stack_trace is " + stack_trace)
            response = jsonify({"result": "NA", "error": "Json Decoding failed. Check if the payload is correct",
                                "exception": str(e), "stack_trace": stack_trace})
            response.status_code = 400
            return response
        model_id = request.args.get('model_id')
        model_version = request.args.get('model_version')
        key = (model_id, model_version)

        try:
            curr_model = models_loaded[key]
        except KeyError:
            app.logger.error("Model: (%s,%s) doesn't exist " %(model_id, model_version))
            response = jsonify({"result":"NA", "stack_trace" : "Model: (%s,%s) doesn't exist. Deploy this model on Hunch " %(model_id, model_version)})
            response.status_code = 400
            return response
        output = curr_model.predict(input)
        response = jsonify({"result":output,"stack_trace":"NA"})
        response.status_code = 200
        return response
    except Exception as e:
        exc_traceback = str(traceback.format_exc())
        app.logger.error("Exception occurred: " + str(e) + "," + exc_traceback)
        response = jsonify({"result":"NA", "stack_trace": exc_traceback})
        response.status_code = 500
        return response


def lock(lockfile):
    import fcntl
    lockfd = open(lockfile, 'w+')
    fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lockfd


def unlock(lockfd):
    import fcntl
    fcntl.flock(lockfd, fcntl.LOCK_UN)

def write_rotation_status(file_path, status):
    if os.path.isfile(file_path):
        with open(file_path) as f:
            lines = f.read().strip()
            if lines == status:
                return
    lockfile = file_path + '.lock'
    if not os.path.exists(lockfile):
        fd = open(lockfile, 'w+')
        fd.close()

    lockfd = lock(lockfile)
    file = open(file_path, 'w+')
    file.write(status)
    file.close()
    unlock(lockfd)
