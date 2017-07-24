#!/usr/bin/env python

import json
import tensorflow as tf

learningRate = 0.01
trainingIteration = 2000
displayStep = 100


def reduceToProducts(listingWords, featureStrings, session, model, productStrings, jsonline, results):
	featureList = []
	for feature in featureStrings:
		if feature in listingWords:
			featureList.append(1)
		else:
			featureList.append(0)
	prediction = session.run(model, { features: [featureList]})
	if len([ x for x in prediction[0] if x > .5 ]) == 1:
		i = session.run(tf.argmax(prediction[0]))
		productString = productStrings[i]
		if results.get(productString):
			results[productString].append(jsonline)
		else:
			results[productString] = [jsonline]

def readProducts():
	products = {}
	featurenames = {}
	numProducts = 0
	numFeatures = 0
	with open('products.txt') as f:
		for line in f:
			jsonline = json.loads(line)
			numProducts += 1
			productName = jsonline[u'product_name']
			products[productName] = jsonline;
			originalWords = [jsonline[u'manufacturer'], jsonline[u'model']]
			if jsonline.get('family'):
				originalWords.append(jsonline[u'family'])
			strippedmodel = jsonline[u'model'].replace(" ", "")
			if jsonline[u'model'] != strippedmodel:
				originalWords.append(strippedmodel)
			strippedmodel2 = jsonline[u'model'].replace("-", "")
			if jsonline[u'model'] != strippedmodel2:
				originalWords.append(strippedmodel2)
			jsonline['originalWords'] = originalWords
			for entry in originalWords:
				words = entry.split()
				for word in words:
					if featurenames.get(word) == None:
						featurenames[word]  = word
						numFeatures += 1
	featureStrings = [] # this the names of the inputs.
	for feature in featurenames:
		featureStrings.append(feature)
	return numProducts, numFeatures, featureStrings, products

def buildProductStrings(numProducts, featureStrings, products):
	productStrings = [] # the names of the products (outputs)
	featuresLists = [] # what's turned on
	productsLists = [] # what should be turned on
	k = 0
	for product,jsonline in products.items():
		originalWords = jsonline[u'originalWords']
		flist = []
		for feature in featureStrings:
			if feature in originalWords:
				flist.append(1)
			else:
				flist.append(0)
		featuresLists.append(flist)
		output = [0] * numProducts
		output[k] = 1
		k += 1
		productsLists.append(output)
		productStrings.append(product)
	return productStrings, featuresLists, productsLists

def createModel(numFeatures, numProducts, features):
	weights = tf.Variable(tf.zeros([numFeatures, numProducts]))
	biases = tf.Variable(tf.ones([numProducts]))

	with tf.name_scope("Wx_b") as scope:
		model = tf.nn.softmax(tf.matmul(features, weights) + biases)
	return model

numProducts, numFeatures, featureStrings, products = readProducts()
productStrings, featuresLists, productsLists = buildProductStrings(numProducts, featureStrings, products)

features = tf.placeholder("float", [None, numFeatures])
outputs = tf.placeholder("float", [None, numProducts])
model = createModel(numFeatures, numProducts, features)


costFunction = tf.square(outputs - model)

optimizer = tf.train.AdamOptimizer(learningRate).minimize(costFunction)

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)

	oldCost = -1

	for iteration in range(trainingIteration):
		fdict = {features: featuresLists, outputs: productsLists }
		session.run(optimizer, fdict)
		if iteration % displayStep == 0:
			avgCost = session.run(tf.reduce_sum(session.run(costFunction, fdict)))
			print ("Iteration:{:4} cost={:.9f}".format(iteration, avgCost))
			predictions = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
			accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
			print("Accuracy:", accuracy.eval({features: featuresLists, outputs: productsLists}))
			if oldCost < 0:
				oldCost = avgCost
			elif oldCost > avgCost:
				oldCost = avgCost
			elif oldCost < avgCost * .9:
				break

	print("done training")

	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print("Accuracy:", accuracy.eval({features: featuresLists, outputs: productsLists}))

	results = {}
	iteration = 0
	with open('listings.txt') as f:
		for line in f:
			jsonline = json.loads(line)
			if jsonline.get('title') != None:
				words = jsonline[u'title']
				listingWords = words.split()
			elif jsonline.get('manufacturer') != None:
				listingWords = [jsonline[u'manufacturer'], jsonline[u'model']]
				if jsonline.get('family'):
					listingWords.append(jsonline[u'family'])
			else:
				listingWords = products[productStrings[0]]['originalWords']
			reduceToProducts(listingWords, featureStrings, session, model, productStrings, jsonline, results)
			iteration += 1
			if iteration % displayStep == 0:
				print("Listing:{:4}".format(iteration))

with open('results.txt', "w") as f:
	for prod,listing in results.items():
		print('{{ "product_name": "{:}",'.format( prod),' "listings":', json.dumps(listing), '}', file=f)

