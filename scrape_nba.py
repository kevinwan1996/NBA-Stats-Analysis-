from urllib2 import urlopen
from bs4 import BeautifulSoup, Comment
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, IndexFormatter
import re
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift


"""gets the counting per game stats of all players in a current year, sort by
any per game counting stat you choose"""

def get_all_players_stats_year(year, sort_by='PS/G'):
	""" will return a dataframe that consists of all the stats of every player for a single year """

	url = 'http://www.basketball-reference.com/leagues/NBA_'+str(year)+'_per_game.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')

	#stat names, PTS, AST ...
	column_headers = [th.getText() for th in soup.find('thead').findAll('tr')[0].findAll('th') if th.getText() != 'Rk']

	#the rows in the table that hold the player data
	data_rows = soup.find('tbody').findAll('tr')

	#store the player data
	player_data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]

	#remove the incomplete data -- mainly the headers in the middle of the table
	for player in player_data:
		if len(player) < 25:
			player_data.remove(player)

	traded_players = []
	for player in player_data:
		if 'TOT' in player:
			traded_players.append(player)
			player_data.remove(player)

	#add all of the teams into team played for instead of just "TOT"
	for traded in traded_players:
		for player in player_data:
			if player[0] == traded[0]:
				if traded[3] == 'TOT':
					traded[3] = player[3]
				else:
					traded[3] = traded[3] + '/' + player[3]

	#merge the lists back together
	player_data = player_data + traded_players


	df = pd.DataFrame(player_data, columns=column_headers)
	# print df.loc[df['PS/G'] == '31.6']

	#convert numeric types to the proper data type
	df = df.convert_objects(convert_numeric=True)
	return df.sort_values(by=[sort_by], ascending=False)

#get a dictionary mapping the player name to the link to their personal page
def dict_of_players(year1, year2):
	dict_of_players = {}
	for year in range(year1, year2):
		url = 'http://www.basketball-reference.com/leagues/NBA_'+str(year)+'_per_game.html'
		html = urlopen(url)
		soup = BeautifulSoup(html, 'lxml')

		#get the rows that hold the data
		data_rows = soup.find('tbody').findAll('tr')

		#get the td tag that holds the 'data-append-csv' tag which is the player link and store in list
		player_links = []
		for i in range(len(data_rows)):
			temp = []
			temp = data_rows[i].findAll('td', {'data-append-csv':True})
			print temp
			player_links.append(temp)

		#put the links into dict with name as key and link as value
		for player in player_links:
			for val in player:
				dict_of_players[val.getText()] = val['data-append-csv']

	# print len(dict_of_players)
	# print dict_of_players['Kobe Bryant']
	return dict_of_players

#will get career per game counting stats
def get_individual_player_career_per_game_stats(player_link):
	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')

	player = soup.find(id="meta").find('h1').getText()

	#header info
	column_headers = [th.getText() for th in soup.find('thead').findAll('tr')[0].findAll('th')]

	#find the rows that hold all of the data
	data_rows = soup.find('tbody').findAll('tr')

	#append all the data to another array and make a df
	player_data = []
	for i in range(len(data_rows)):
		player_row = []
		if data_rows[i].find('th') == None:
			continue
		player_row.append(data_rows[i].find('th').getText())
		# player_data.append(data_rows[i].findAll('td').getText())
		for td in data_rows[i].findAll('td'):
			player_row.append(td.getText())
		player_data.append(player_row)

	df = pd.DataFrame(player_data, columns=column_headers)
	df = df.convert_objects(convert_numeric=True)
	return df, player

#get the other commented out tables
"""includes advanced stats, playoff stats, total stats, ..."""
def get_table_from_comments(player_link, div_selector):
	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')
	comments = soup.findAll(text=lambda text:isinstance(text, Comment))
	# [comment.extract() for comment in comments]

	player = soup.find(id="meta").find('h1').getText()

	#look through the comments for the proper table, create header and data df
	for comment in comments:
			comment_soup = BeautifulSoup(comment, 'lxml')
			if comment_soup.find(id=div_selector) != None:
				column_headers = []
				for th in comment_soup.find('thead').findAll('tr')[0].findAll('th'):
					# if th.getText() != None:
					column_headers.append(th.getText())
				data_rows = comment_soup.find('tbody').findAll('tr')
				player_data = []
				for i in range(len(data_rows)):
					player_row = []
					player_row.append(data_rows[i].find('th').getText())
					for td in data_rows[i].findAll('td'):
						# if td.getText() != '':
						player_row.append(td.getText())
					player_data.append(player_row)

	df = pd.DataFrame(player_data, columns=column_headers)
	df = df.convert_objects(convert_numeric=True)
	return df, player

"""neccessary for shooting, play-by-play, playoffs shooting, playoffs play-by-play,
similarity scores, and college"""
def get_multi_header_table(player_link, div_selector):
	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')
	comments = soup.findAll(text=lambda text:isinstance(text, Comment))

	player = soup.find(id="meta").find('h1').getText()

	#look through comments for table, create df w/header[1] and data
	for comment in comments:
		comment_soup = BeautifulSoup(comment, 'lxml')
		if comment_soup.find(id=div_selector) != None:
			column_headers = []
			for th in comment_soup.find('thead').findAll('tr')[1].findAll('th'):
			# if th.getText() != None:
				column_headers.append(th.getText())
			data_rows = comment_soup.find('tbody').findAll('tr')
			player_data = []
			for i in range(len(data_rows)):
				player_row = []
				player_row.append(data_rows[i].find('th').getText())
				for td in data_rows[i].findAll('td'):
					# if td.getText() != '':
					player_row.append(td.getText())
				player_data.append(player_row)

	df = pd.DataFrame(player_data, columns=column_headers)
	df = df.convert_objects(convert_numeric=True)
	# df[''].plot(title=div_selector[4:])
	# plt.show()
	return df, player

def plot_dataframe(df, statistic, plot_name, x_label, y_label, x_ticks, plot_type='bar'):
	#choose the column to plot, title of plot, and kind of plot

	fig = df[statistic].plot(title=plot_name, kind=plot_type)
	#set the max number of x-ticks and format them 

	fig.xaxis.set_major_locator(MaxNLocator(len(df)))
	fig.xaxis.set_major_formatter(IndexFormatter(df.index))
	#display grid

	fig.grid(which='minor', alpha=0.2)
	fig.grid(which='major', alpha=0.5)
	#set x and y labels

	fig.set_xlabel(x_label)
	fig.set_ylabel(y_label)
	#set the labels based on the column chosen

	fig.set_xticklabels(df[x_ticks], rotation=30)
	for tick in fig.xaxis.get_majorticklabels():
		tick.set_horizontalalignment("right")

	# fig.get_figure().savefig("/Users/Kevin/Documents/work/python_bot/plot.jpg")
	#show the plot
	plt.tight_layout()
	plt.show()

#plot the stats of two players versus each other
def compare_players(df1, df2, p1, p2, title, statistic):
	fig = df1[statistic].plot(title=title, legend=True)
	df2[statistic].plot(ax=fig, legend=True)
	L = plt.legend()
	L.get_texts()[0].set_text(p1)
	L.get_texts()[1].set_text(p2)
	plt.show()

#get all personal info including player name, height/weight, birthplace, etc...
def get_info(player_link):

	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')		

	info = soup.find(id="meta")
	# for p in info.findAll('p'):
	# 	print p.getText()
	player_info = {}
	#get name
	player_info['Name'] = info.find('h1').getText()

	#acceptable characters for key/value
	whitelist = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ')
	calendar_list = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890')

	#Position and Shoots
	# position = info.findAll('p')[2].findAll('strong')
	# for strong_tag in position:
		# player_info[''.join(filter(whitelist.__contains__,strong_tag.getText().strip()))] = ''.join(filter(whitelist.__contains__, strong_tag.next_sibling.strip()))

	p_tags = info.findAll('p')
	for p in p_tags:
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Position':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].next_sibling.strip()))
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Pronunciation':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].next_sibling.strip()))
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Born':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = p.findAll('span')[0].findAll('a')[0].getText() + " " + p.findAll('span')[0].findAll('a')[1].getText()
			birth_place = ''.join(filter(whitelist.__contains__, p.findAll('span')[1].getText().strip()))
			birth_place = birth_place[2:]
			birth_place = re.sub(r"\B([A-Z])", r" \1", birth_place)
			player_info['Birthplace'] = birth_place
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Died':
			death_day = ''.join(filter(calendar_list.__contains__, p.findAll('span')[0].getText()))
			death_day = re.sub(r"\B([A-Z])", r" \1", death_day).strip()
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = death_day[0:-6] + ' ' + death_day[-6:-4] + ', ' + death_day[-4:]
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'College':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = p.find('a').getText()
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'High School':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].next_sibling.strip()))
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Draft':
			player_info['Draft Team'] = ''.join(filter(whitelist.__contains__, p.findAll('a')[0].getText().strip()))
			player_info['Draft Year'] = ''.join(filter(whitelist.__contains__, p.findAll('a')[1].getText().strip()))
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'NBA Debut':
			debut = ''.join(filter(calendar_list.__contains__, p.findAll('a')[0].getText()))
			debut = re.sub(r"\B([A-Z])", r" \1", debut).strip()
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = debut[0:-7] + debut[-7:-4] + ', ' + debut[-4:]
		if len(p.findAll('strong')) > 0 and ''.join(filter(whitelist.__contains__, p.findAll('strong')[0].getText().strip())) == 'Hall of Fame':
			player_info[''.join(filter(whitelist.__contains__,p.findAll('strong')[0].getText().strip()))] = ''.join(filter(calendar_list.__contains__, p.findAll('strong')[0].next_sibling.strip()))

	return player_info

#return the player's nicknames as listed on BBRef
def get_nicknames(player_link):
	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')	

	return soup.find(id="meta").findAll('p')[1].getText()

#get the accolades the player has won
def get_accolades(player_link):
	url = 'http://www.basketball-reference.com/players/'+player_link[0:1]+'/'+player_link+'.html'
	html = urlopen(url)
	soup = BeautifulSoup(html, 'lxml')	
	
	accomplishments = soup.find(id="bling")

	list_of_accolades = []
	for li in accomplishments.findAll('li'):
		list_of_accolades.append(li.find('a').getText())
	return list_of_accolades


#takes too long to generate a dictionary every run, so store links and names into a Database
def create_link_name_db(path_to_db, player_dict, table_name, column_links='LINK', column_name='NAME'):
	conn = sqlite3.connect(path_to_db)
	c = conn.cursor()
	c.execute("create table {name} ({column} TEXT)".format(name=table_name, column=column_name))
	c.execute("alter table {name} add column '{column}' TEXT".format(name=table_name, column=column_links))
	
	for key in player_dict:
		c.execute("Insert Into {name} ({column_1}, {column_2}) Values (?, ?)".format(name=table_name, column_1=column_name, column_2=column_links), [key.lower(), player_dict[key]])
	conn.commit()
	conn.close()

#query the database for the player link
def get_player_link(path_to_db, player_name, table_name, column_links='LINK', column_name='NAME'):
	conn = sqlite3.connect(path_to_db)
	c = conn.cursor()
	c.execute("Select {column_links} from {table_name} where {column_name}= ? ".format(table_name=table_name, column_name=column_name, column_links=column_links), [player_name.lower()])
	all_rows = c.fetchall()[0]	
	return (all_rows)[0]

#workaround to find dataframe by position
def position_dataframe(df, year, position):
	if 'g' in position.lower():
		df = df.drop(df[df['Pos'] == 'SF'].index)
		df = df.drop(df[df['Pos'] == 'PF'].index)
		df = df.drop(df[df['Pos'] == 'SF-PF'].index)
		df = df.drop(df[df['Pos'] == 'PF-SF'].index)
		df = df.drop(df[df['Pos'] == 'PF-C'].index)
		df = df.drop(df[df['Pos'] == 'C'].index)
		df = df.drop(df[df['Pos'] == 'C-PF'].index)
	if 'f' in position.lower():
		df = df.drop(df[df['Pos'] == 'PG'].index)
		df = df.drop(df[df['Pos'] == 'SG'].index)
		df = df.drop(df[df['Pos'] == 'PG-SG'].index)
		df = df.drop(df[df['Pos'] == 'SG-PG'].index)
		df = df.drop(df[df['Pos'] == 'C'].index)
	if 'c' in position.lower():
		df = df.drop(df[df['Pos'] == 'PG'].index)
		df = df.drop(df[df['Pos'] == 'SG'].index)
		df = df.drop(df[df['Pos'] == 'SF'].index)
		df = df.drop(df[df['Pos'] == 'PF'].index)
		df = df.drop(df[df['Pos'] == 'PG-SG'].index)
		df = df.drop(df[df['Pos'] == 'SG-PG'].index)
		df = df.drop(df[df['Pos'] == 'PF-SS'].index)
		df = df.drop(df[df['Pos'] == 'SF-PF'].index)
	
	return df

#get a dataframe consisting of only 2 specific statistics
def statistic_df(df, year, statistic_1, statistic_2):
	df = df[['Player', statistic_1, statistic_2]]
	df = df.drop(df[df[statistic_1].isnull()].index)
	df = df.drop(df[df[statistic_2].isnull()].index)
	return df

#get dataframe of two specific stats for one position ...SLOW right now
def position_statistic_df(year, statistic_1, statistic_2, position):
	df = get_all_players_stats_year(year, 'PS/G')
	df = position_dataframe(df, year, position)
	df = statistic_df(df, year, statistic_1, statistic_2)
	return df

#convert a dataframe to a numpy array
def df_to_array(df):
	return df.values

#find clusters based on distance away from centroids
def KMeans_classify(array,clusters):
	#separate the player name from the statistic
	players = []
	stats = []
	for val in array:
		players.append(val[0])
		temp_stats = []
		temp_stats.append(val[1])
		temp_stats.append(val[2])
		stats.append(temp_stats)
	
	#create the classifier
	clf = KMeans(n_clusters=clusters)
	clf.fit(stats)
	
	#get the centroids and labels
	centroids = clf.cluster_centers_
	labels = clf.labels_

	player_labels = {}
	for i in range(len(labels)):
		player_labels[players[i]] = labels[i]

	colors = ['g.', 'r.', 'c.', 'y.']
	for i in range(len(stats)):
		plt.plot(stats[i][0], stats[i][1], colors[labels[i]], markersize = 10)
		# plt.annotate(players[i], xy=(stats[i][0], stats[i][1]))
	plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
	plt.show()

	return player_labels

def Mean_Shift_classify(array):
	players = []
	stats = []
	for val in array:
		players.append(val[0])
		temp_stats = []
		temp_stats.append(val[1])
		temp_stats.append(val[2])
		stats.append(temp_stats)
	
	#create classifier
	ms = MeanShift()
	ms.fit(stats)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	player_labels = {}
	for i in range(len(labels)):
		player_labels[players[i]] = labels[i]

	print(cluster_centers)
	n_clusters_ = len(np.unique(labels))
	print("Number of estimated clusters:", n_clusters_)

	colors = 10*['r','g','b','c','k','y','m']
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for i in range(len(stats)):
		ax.scatter(stats[i][0], stats[i][1], c=colors[labels[i]], marker='o')
	ax.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)
	plt.show()
	
	return player_labels
# def numpy_statistic_array(year, statistic_1, statistic_2, position):
if __name__=='__main__':
	# get_player_stats_year(2017, 'PS/G')
	# dict_of_players()
	# data, player = get_table_from_comments('bryanko01','div_advanced')
	# data, player = get_individual_player_career_per_game_stats('bryanko01')
	# print data
	# data2, player2 = get_table_from_comments('curryst01', 'div_advanced')
	# compare_players(data,data2, player, player2, 'Kobe vs. Curry', 'VORP')
	# print get_individual_player_career_per_game_stats('curryst01')
	# get_multi_header_table('curryst01', 'div_playoffs_advanced_pbp')
	# data = get_all_players_stats_year(2006, 'PS/G').head()
	# db_location = '/Users/Kevin/Desktop/NBA_links_names.sqlite'
	# print get_accolades('jordami01')
	# d = dict_of_players(2014,2015)
	# create_link_name_db('/Users/Kevin/Desktop/test_nba.sqlite', d, 'PlayerLinks')
	# a =  get_player_link('/Users/Kevin/Desktop/test_nba.sqlite', 'pAUl GeoRGE', 'PlayerLinks')
	# print a
	# data, player = get_individual_player_career_per_game_stats(a)
	# print data
	# plot_dataframe(data, 'PTS', 'Paul George PTS', 'Season', 'PTS', 'Season', 'line')
	# df = position_dataframe(2015, 'Guard')
	# print statistic_array(2015, 'AST', 'TOV') 
	style.use('ggplot')
	# X = df_to_array(position_statistic_df(2015, 'AST', 'TOV', 'Guard'))

	# Y = df_to_array(position_statistic_df(2015, 'FG%', 'PS/G', 'Guard'))
	Z = df_to_array(position_statistic_df(2015, '3P%', '3P', 'Guard'))
	# W = df_to_array(position_statistic_df(2015, 'ORB', 'DRB', 'Guard'))

	# plt.scatter(X[:, 1],X[:, 2], s=15, c = 'b', alpha=0.7, linewidths = 5, zorder = 10)
	# plt.show()
	# threes = KMeans_classify(Z, 4)
	threes = Mean_Shift_classify(Z)
	print threes['Stephen Curry']



