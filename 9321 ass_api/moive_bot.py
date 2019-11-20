from flask import request,jsonify
from flask import Flask, render_template
from api_test import detect_intent_texts
from flask_restful import Resource, Api,fields
import  requests
import pymysql


def after_request(response):
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
	response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
	return response
	

def get_movie_info(movie_name):
	db = pymysql.connect(host='database-1.clr3d8nnckz4.us-east-2.rds.amazonaws.com', user='admin', password='19950423',port=3306, db='Movie')
	cursor = db.cursor()
	response = ''
	sql_query = "SELECT Description FROM movie where Title = %s"
	cursor.execute(sql_query, movie_name)
	result = cursor.fetchall()
	db.commit()
	if not result:
		response += f"Sorry,there is no movie called {movie_name},please check it"
	else:
		result = result[0][0]
		response += str(result)
	return response
def movie_recommed(movie_type):
	response = ''
	movie_type_1 = "%" + movie_type + "%"
	db = pymysql.connect(host='database-1.clr3d8nnckz4.us-east-2.rds.amazonaws.com', user='admin', password='19950423',
						 port=3306, db='Movie')
	cursor = db.cursor()
	response = ''
	sql_query = "SELECT Title FROM movie where Genre like %s order by Rating DESC "
	cursor.execute(sql_query, movie_type_1)
	result = cursor.fetchall()
	response += f"Here are top rating {movie_type} movie for you:\n"
	if len(result) > 5:
		for i in range(5):
			response += (result[i][0])
			response += "\n"
	else:
		for i in range(len(result)):
			response += (result[i][0])
			response += "\n"
	print(response)
	return response
def recommend_top_movie():
	response = ''
	db = pymysql.connect(host='database-1.clr3d8nnckz4.us-east-2.rds.amazonaws.com', user='admin', password='19950423',
						 port=3306, db='Movie')
	cursor = db.cursor()
	response = 'Here are top10 rating movies for you:\n'
	sql_query = "SELECT Title FROM movie order by Metascore DESC "
	cursor.execute(sql_query)
	result = cursor.fetchall()
	for i in range(10):
		response += (result[i][0])
		response += "\n"
	print(response)
	return response
	
app = Flask(__name__)
app.after_request(after_request)
api = Api(app)


# 输出部分
class moive_bot(Resource):
	#
	def post(self,session_id,user_input):
		response = detect_intent_texts(session_id, user_input)
		intent = response.query_result.intent.display_name
		data = response.query_result.parameters
		if intent == 'Default Fallback Intent' :
			return "I am quite confused"
		if "Movie_name" in data :
			movie_name = data["Movie_name"].replace("<","").replace(">","")
			response = get_movie_info(movie_name)
			return response,200
		if "movie_type" in data :
			movie_type = data["movie_type"]
			response = movie_recommed(movie_type)
			return response, 200
		if "recommend_part" in data:
			response = recommend_top_movie()
			return response,200

		return response.query_result.fulfillment_text,200
@app.route('/movie', methods=['POST'])
def OMDB_API():
	apikey = 'ce40ddb8'
	title_search = request.form['title_search']
	print(title_search)
	r = requests.get('http://www.omdbapi.com/?apikey='+apikey+'&s='+title_search)
	json_object = r.json()

	items = json_object['Search']

	for item in items:
		title = item['Title']
		year = item['Year']
		poster = item['Poster']
		imdbID = item['imdbID']

	return render_template('movie.html', items=items)

@app.route('/main')
def index():
	return render_template('index.html')

				
api.add_resource(moive_bot,'/<string:session_id>/<string:user_input>')






if __name__ == "__main__":
	app.run(debug=True)
	
