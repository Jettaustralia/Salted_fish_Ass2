import dialogflow_v2beta1 as dialogflow
import os


def detect_intent_texts(session_id, texts, language_code='en-US',project_id='movie-qijbmr'):
	path = os.path.join((os.path.dirname(os.path.abspath('api_test.py'))),
						'Movie-a8de2ddc64b9.json')
#	print(path)
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
	session_client = dialogflow.SessionsClient()

	session = session_client.session_path(project_id, session_id)
	print('Session path: {}\n'.format(session))
	text_input = dialogflow.types.TextInput(
		text=texts, language_code=language_code)
	query_input = dialogflow.types.QueryInput(text=text_input)
	response = session_client.detect_intent(
		session=session, query_input=query_input)
		
	return response
# def print_res(response):
# #	print('=' * 20)
# #	print('Query text: {}'.format(response.query_result.query_text))
# #	print('Detected intent: {} (confidence: {})\n'.format(
# #		response.query_result.intent.display_name,
# #		response.query_result.intent_detection_confidence))
# 	print('{}\n'.format(
# 		response))


if __name__ == '__main__':
	res=detect_intent_texts('2', 'Please recommend a thriller movie!', 'en-US','movie-qijbmr')
#	print_res(res.query_result.parameters.fields)
	print(res.query_result.parameters["movie_type"])
#	if 'Movie_name' in res.query_result.parameters:
#		print(1)
