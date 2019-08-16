#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "HyperParams.h"
#include <boost/range/adaptor/reversed.hpp>


struct GraphBuilder {
    DynamicLSTMBuilder target_encoder_l2r;
    DynamicLSTMBuilder target_encoder_r2l;

    DynamicLSTMBuilder words_encoder_l2r;
    DynamicLSTMBuilder words_encoder_r2l;

    std::vector<Node *>target_encoder_lookups;
    std::vector<Node *>words_encoder_lookups;

    BucketNode *hidden_bucket = new BucketNode;

    std::vector<ConcatNode *> target_concats;
    std::vector<ConcatNode *> words_concats;

    MaxPoolNode *target_pool = new MaxPoolNode;
    MaxPoolNode *words_pool = new MaxPoolNode;

    UniNode * neural_output = new UniNode;

    UniNode *forward(Graph &graph, ModelParams &model_params, HyperParams &hyper_params, 
                const Feature &feature, bool is_trainning) {
        
        hidden_bucket -> init(hyper_params.hiddenSize);
        hidden_bucket -> forward(graph);

        for (const string &word : feature.m_target) {
                LookupNode *target_lookup = new LookupNode;
                target_lookup -> init(hyper_params.wordDim);
                target_lookup -> setParam(model_params.words);
                target_lookup -> forward(graph, word);

                DropoutNode *dropout_node = new DropoutNode(hyper_params.dropProb, is_trainning);
                dropout_node -> init(hyper_params.wordDim);
                dropout_node -> forward(graph, *target_lookup);

                target_encoder_lookups.push_back(dropout_node);
        }

        for (Node *node : target_encoder_lookups) {
            target_encoder_l2r.forward(graph, model_params.lstm_target_left_params, *node, *hidden_bucket, *hidden_bucket, hyper_params.dropProb, is_trainning);
        }

        for (const string &word : feature.m_words) {
                LookupNode *words_lookup = new LookupNode;
                words_lookup -> init(hyper_params.wordDim);
                words_lookup -> setParam(model_params.words);
                words_lookup -> forward(graph, word);

                DropoutNode *dropout_node = new DropoutNode(hyper_params.dropProb, is_trainning);
                dropout_node -> init(hyper_params.wordDim);
                dropout_node -> forward(graph, *words_lookup);

                words_encoder_lookups.push_back(dropout_node);
        }

        Node *hidden_c_l2r = target_encoder_l2r._cells.at(target_encoder_lookups.size() - 1);

        for (Node *node : words_encoder_lookups) {
            words_encoder_l2r.forward(graph, model_params.lstm_tweet_left_params, *node, *hidden_bucket, *hidden_c_l2r, hyper_params.dropProb, is_trainning);
        }

        for (Node *node : boost::adaptors::reverse(target_encoder_lookups)) {
            target_encoder_r2l.forward(graph, model_params.lstm_target_right_params, *node, *hidden_bucket, *hidden_bucket, hyper_params.dropProb, is_trainning);
        }

        Node *hidden_c_r2l = target_encoder_r2l._cells.at(target_encoder_lookups.size() - 1);

        for (Node *node : boost::adaptors::reverse(words_encoder_lookups)) {
            words_encoder_r2l.forward(graph, model_params.lstm_tweet_right_params, *node, *hidden_bucket, *hidden_c_r2l, hyper_params.dropProb, is_trainning);
        }

        for (int i = 0; i < feature.m_target.size(); i++) {
            ConcatNode *concat(new ConcatNode);
            concat -> init(hyper_params.hiddenSize * 2);
            concat -> forward(graph, {target_encoder_l2r._hiddens.at(i), target_encoder_r2l._hiddens.at(feature.m_target.size() - (i + 1))});
            target_concats.push_back(concat);
        }

        for (int i = 0; i < feature.m_words.size(); i++) {
            ConcatNode *concat = new ConcatNode;
            concat -> init(hyper_params.hiddenSize * 2);
            concat -> forward(graph, {words_encoder_l2r._hiddens.at(i), words_encoder_r2l._hiddens.at(feature.m_words.size() - (i + 1))});
            words_concats.push_back(concat);
        }


        target_pool -> init(hyper_params.hiddenSize * 2);
        target_pool -> forward(&graph,  toNodePointers<ConcatNode>(target_concats));

        words_pool -> init(hyper_params.hiddenSize * 2);
        words_pool -> forward(&graph, toNodePointers<ConcatNode>(words_concats));

        ConcatNode *pool_concat = new ConcatNode;
        pool_concat -> init(hyper_params.hiddenSize * 4);
        pool_concat -> forward(graph, {target_pool, words_pool});

        neural_output -> setParam(&model_params.olayer_linear);
        neural_output -> init(hyper_params.labelSize);
        neural_output -> forward(graph, *pool_concat);

        return neural_output;
    }
};


#endif /* SRC_ComputionGraph_H_ */
