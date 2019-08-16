/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

//A native neural network classfier using only word embeddings

class Driver{
	public:
		Driver(int memsize) {

		}

		~Driver() {

		}

	public:
		ModelParams _modelparams;  // model parameters
		HyperParams _hyperparams;
		Metric _eval;
        vector<GraphBuilder> _builders;
		vector<Node*> _output;

		ModelUpdate _ada;  // model update

	public:
		//embeddings are initialized before this separately.
		inline void initial() {
			if (!_hyperparams.bValid()){
				std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
				return;
			}
			if (!_modelparams.initial(_hyperparams)){
				std::cout << "model parameter initialization Error, Please check!" << std::endl;
				return;
			}
			_modelparams.exportModelParams(_ada);
			//_modelparams.exportCheckGradParams(_checkgrad);

			_hyperparams.print();


			setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		}




		inline dtype train(Graph& graph, const vector<Example>& examples, int iter) {
			_eval.reset();
		    _builders.clear();
		    _output.clear();
                // UniNode* _O;
			Node *result;
			dtype cost = 0.0;
			int example_num = examples.size();
			for (int count = 0; count < example_num; count++) {
				const Example& example = examples.at(count);
				//forward
				GraphBuilder builder;
				_builders.push_back(builder);
				result = _builders.at(count).forward(graph, _modelparams, _hyperparams, example.m_feature, true);
                _output.push_back(result);
			}
			// std::cout << "************************************" << _output.size() << std::endl;
			// std::cout << "------------------------------------" <<_output.at(0) -> getDim() << std::endl;
			// std::cout << "------------------------------------" <<_builders.size() << std::endl;
			graph.compute();

			//
			
			for (int count = 0; count < example_num; count++) {
				const Example& example = examples.at(count);
				cost += softMaxLoss(_output.at(count), example.m_label, _eval, example_num);
			}

			graph.backward();

			if (_eval.getAccuracy() < 0) {
				std::cout << "strange" << std::endl;
			}

			return cost;
		}

		inline void predict(Graph& graph, const Feature& feature, int& result) {
        	 Node* _P;
			 GraphBuilder builder;
			_P = builder.forward(graph, _modelparams, _hyperparams, feature, false);
			graph.compute();
			bool bTargetInTweet = IsTargetIntweet(feature);
			predictLoss(_P, result, bTargetInTweet);
		}

		inline bool IsTargetIntweet(const Feature& feature) {
			string words = "";
			for (int i = 0; i < feature.m_words.size(); i++)
				words = words + feature.m_words[i];
			string::size_type idx;
			if (feature.m_target[0] == "hillary") {
				idx = words.find("hillary");
				if (idx != string::npos) return true;
				idx = words.find("clinton");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "trump") {
				idx = words.find("trump");
				if (idx != string::npos) return true;
				idx = words.find("donald");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "climate") {
				idx = words.find("climate");
				if (idx != string::npos) return true;
			}
			if (feature.m_target[0] == "feminism") {
				idx = words.find("feminism");
				if (idx != string::npos) return true;
				idx = words.find("feminist");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "abortion") {
				idx = words.find("abortion");
				if (idx != string::npos) return true;
				idx = words.find("aborting");
				if (idx != string::npos) return true;

			}
			if (feature.m_target[0] == "atheism") {
				idx = words.find("atheism");
				if (idx != string::npos) return true;
				idx = words.find("atheist");
				if (idx != string::npos) return true;

			}
			return false;
		}


		void updateModel() {
			//_ada.update();
			//_ada.update(5.0);
			//_ada.update(10);
			_ada.updateAdam(10);
		}

		void checkgrad(const vector<Example>& examples, int iter){
			ostringstream out;
			out << "Iteration: " << iter;
		}




	private:
		inline void resetEval() {
			_eval.reset();
		}


		inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
			_ada._alpha = adaAlpha;
			_ada._eps = adaEps;
			_ada._reg = nnRegular;
		}

};

#endif /* SRC_Driver_H_ */
