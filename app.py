from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import pandas as pd

app = Flask(__name__)

# Load Match Prediction Model
pipe_match = pickle.load(open('model.pkl', 'rb'))  # Match Prediction Model
pipe_score = pickle.load(open('cricket_world_cup_score_prediction_model.pkl', 'rb'))  # Score Prediction Model

# Load player statistics
batting_stats = pd.read_pickle("player_stats1.pkl")  # Batting stats
bowling_stats = pd.read_pickle("bowl_stats.pkl")  # Bowling stats

# Define Teams and Cities
teams = [
    'India', 'Pakistan', 'Australia', 'England', 'South Africa',
    'Sri Lanka', 'Bangladesh', 'Afghanistan', 'Netherlands', 'West Indies', 'New Zealand'
]

cities = [
    'Dhaka', 'Chandigarh', 'Colombo', 'Johannesburg', 'London', 'Centurion',
    'Potchefstroom', 'Southampton', 'Bloemfontein', 'Cardiff', 'Lahore', 'Kandy',
    'Hambantota', 'Chattogram', 'Harare', 'Bulawayo', 'Karachi', 'Rawalpindi',
    'Benoni', 'Hamilton', 'Auckland', 'Chennai', 'Visakhapatnam', 'Mumbai',
    'Kimberley', 'Indore', 'Raipur', 'Hyderabad', 'Thiruvananthapuram', 'Kolkata',
    'Guwahati', 'Sydney', 'Adelaide', 'Delhi', 'Ranchi', 'Lucknow', 'Cairns',
    'Rotterdam', 'Manchester', 'Chester-le-Street', 'Amstelveen', 'Mount Maunganui',
    'Doha', 'Cape Town', 'Paarl', 'Birmingham', 'Pune', 'Wellington', 'Christchurch',
    'Dunedin', 'Canberra', 'Bengaluru', 'Rajkot', 'Leeds', 'Nottingham', 'Taunton',
    'Bristol', 'Dubai', 'Abu Dhabi', 'Sharjah', 'Port Elizabeth', 'Nagpur', 'Napier',
    'Durban', 'Melbourne', 'Nelson', 'Hobart', 'Brisbane', 'Dharamsala', 'Kanpur',
    'East London', 'Dublin', 'Cuttack', 'Perth', 'Chittagong', 'Mirpur', 'St Kitts',
    'Guyana', 'Ahmedabad', 'Fatullah', 'Bangalore', 'Jaipur', 'Trinidad', 'Jamaica',
    'Kochi', 'Vadodara', 'Gwalior', 'Darwin', 'Faisalabad', 'Belfast', 'St Lucia',
    'Grenada', 'Barbados', 'Antigua', 'Margao', 'Kuala Lumpur', 'Jamshedpur',
    'Faridabad', 'Bogra', 'Queenstown', 'Canterbury', 'Dambulla', 'Peshawar',
    'Multan', 'Gqeberha', 'Port Moresby', 'Lauderhill', 'Bermuda', 'St Vincent',
    'St Lucia', 'Bridgetown', 'Kingston', 'Grenada', 'Antigua', 'Guyana',
]
cities = sorted(cities)
# Route to serve images
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('images', filename)

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Match Prediction Page
@app.route('/match_prediction')
def match_prediction():
    return render_template('index.html', teams=teams, cities=cities)

# Match Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    venue = request.form['venue']
    score = int(request.form['score'])
    current_score = int(request.form['current_score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

    runs_left = score - current_score
    balls_left = 300 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [venue],
        'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets_left],
        'total_runs_x': [score], 'crr': [crr], 'rrr': [rrr]
    })

    result = pipe_match.predict_proba(input_df)
    loss = result[0][0]
    gain = result[0][1]

    return render_template('result.html', batting_team=batting_team, bowling_team=bowling_team,
                           gain=round(gain * 100), loss=round(loss * 100))

# Score Prediction Page
@app.route('/score_prediction', methods=['GET', 'POST'])
def score_prediction():
    if request.method == 'POST':
        try:
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            current_score = int(request.form['current_score'])
            runs_last_5 = int(request.form['last_five'])
            wickets = int(request.form['wickets'])
            wickets_last_5 = int(request.form['wickets_last_five'])
            overs = float(request.form['overs'])  # <- updated field name
            city = request.form['city']

            input_df = pd.DataFrame({
                'ball': [overs],
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'innings_runs': [current_score],
                'runs_last_5_overs': [runs_last_5],
                'Innings_wickets': [wickets],
                'wickets_last_5_overs': [wickets_last_5],
                'city': [city]
            })

            prediction = pipe_score.predict(input_df)[0]
            prediction = int(round(prediction))

            # Return JSON for AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'prediction': prediction})

            return render_template('score.html', teams=teams, cities=cities, prediction=prediction)
        
        except Exception as e:
            print("Error occurred during prediction:", e)
            return jsonify({'error': str(e)}), 500

    return render_template('score.html', teams=teams, cities=cities, prediction=None)

@app.route('/stats')
def stats():
    print("Stats page is being rendered!")  # Debugging
    return render_template('stats.html', teams=teams)

@app.route('/bowl')
def bowl():
    return render_template('bowl.html', teams=teams)

# Player Statistics APIs
@app.route('/api/player_stats', methods=['GET'])
def get_player_stats():
    team = request.args.get('team', '')

    if team:
        filtered_stats = batting_stats[batting_stats['team'] == team]
    else:
        filtered_stats = batting_stats

    sorted_stats = filtered_stats.sort_values(by='total_runs', ascending=False)
    player_data = sorted_stats[['player', 'total_runs', 'avg_runs', 'avg_strikerate', 'total_4s', 'total_6s', 'innings_count']].to_dict(orient='records')

    return jsonify(player_data)

@app.route('/api/bowl_stats', methods=['GET'])
def get_bowl_stats():
    team = request.args.get('team', '')

    if team:
        filtered_stats = bowling_stats[bowling_stats['team'] == team]
    else:
        filtered_stats = bowling_stats

    sorted_stats = filtered_stats.sort_values(by='total_wicket', ascending=False)
    player_data = sorted_stats[['player', 'total_wicket', 'total_runs1', 'avg_runrate', 'total_overs', 'total_maiden', 'matches']].to_dict(orient='records')

    return jsonify(player_data)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)