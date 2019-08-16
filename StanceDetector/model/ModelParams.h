#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"
#include "HyperParams.h"

struct ModelParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
        Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet targetAlpha;
	//LSTM1Params lstm_left_params;
	LSTM1Params lstm_target_left_params;
	LSTM1Params lstm_target_right_params;
	//LSTM1Params lstm_right_params;
	LSTM1Params lstm_tweet_left_params;
	LSTM1Params lstm_tweet_right_params;

	UniParams olayer_linear; // output

public:
	bool initial(HyperParams& opts){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0){
			return false;
		}
		opts.wordDim = words.nDim;

		lstm_target_left_params.init(opts.hiddenSize, opts.wordDim);
		lstm_target_right_params.init(opts.hiddenSize, opts.wordDim);
		lstm_tweet_left_params.init(opts.hiddenSize, opts.wordDim);
		lstm_tweet_right_params.init(opts.hiddenSize, opts.wordDim);

		//opts.inputSize = opts.hiddenSize;
		opts.inputSize = opts.hiddenSize * 4;
		//opts.inputSize = opts.hiddenSize * 4;
		//opts.inputSize = opts.hiddenSize * 2;
		//opts.inputSize = opts.hiddenSize;
		//opts.inputSize = opts.wordDim;
		olayer_linear.init(opts.labelSize, opts.inputSize, false);

		return true;
	}
        void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		lstm_target_left_params.exportAdaParams(ada);
		lstm_target_right_params.exportAdaParams(ada);
		lstm_tweet_left_params.exportAdaParams(ada);
		lstm_tweet_right_params.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


    Json::Value toJson() const override {
        Json::Value json;
        json["wordAlpha"] = wordAlpha.toJson();
        json["words"] = words.toJson();
        json["targetAlpha"] = targetAlpha.toJson();
        json["lstm_target_left_params"] = lstm_target_left_params.toJson();
        json["lstm_target_right_params"] = lstm_target_right_params.toJson();
        json["lstm_tweet_left_params"] = lstm_tweet_left_params.toJson();
        json["lstm_tweet_right_params"] = lstm_tweet_right_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        wordAlpha.fromJson(json["wordAlpha"]);
        words.fromJson(json["words"]);
        targetAlpha.fromJson(json["targetAlpha"]);
        lstm_target_left_params.fromJson(json["lstm_target_left_params"]);
        lstm_target_right_params.fromJson(json["lstm_right_right_params"]);
        lstm_tweet_left_params.fromJson(json["lstm_tweet_left_params"]);
        lstm_tweet_right_params.fromJson(json["lstm_tweet_right_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&wordAlpha,&words, &targetAlpha, &lstm_target_left_params
             &lstm_target_right_params, &lstm_tweet_left_params, &lstm_tweet_right_params };
    }
#endif
};

#endif
 /* SRC_ModelParams_H_ */
