#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include "Utf.h"
#include "Instance.h"
using namespace std;
class Reader
{
public:
	Reader()
	{
	}

	virtual ~Reader()
	{
		if (m_inf.is_open()) m_inf.close();
	}
	int startReading(const char *filename) {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
		m_inf.open(filename);

    if (!m_inf.is_open()) {
			cout << "Reader::startReading() open file err: " << filename << endl;
			return -1;
		}

		return 0;
	}

	void finishReading() {
		if (m_inf.is_open()) {
			m_inf.close();
			m_inf.clear();
		}
	}

	virtual Instance *getNext() = 0;
protected:
	ifstream m_inf;

	int m_numInstance;

	Instance m_instance;
};
vector<string> readLines(const string &fullFileName) {
	vector<string> lines;
	std::ifstream input(fullFileName);
	for (std::string line; getline(input, line);) {
		lines.push_back(line);
	}
	return lines;
}

void readLineToInstance_post(const string &line, Instance *instance) {

	vector<string>vec;
	vec.clear();
	string sub = "";
	bool is_space = false;
	for (int i = 0; i < line.length(); i++) {
		if (line[i] == ' ' || line[i] == 9) {
			if (is_space) continue;
			vec.push_back(sub);
			sub = "";
			is_space = true;
			continue;
		}
		sub = sub + line[i];
		is_space = false;
	}
	vec.push_back(sub);
	vector<string>::iterator k = vec.begin();
        vec.erase(k);
        k = vec.begin();
        vec.erase(k);
	for (int i = 0; i < vec.size();i++) {
            string word = normalize_to_lowerwithdigit(vec[i]);
         //   if (word[0] == '@') continue;
	    instance->m_target.push_back(word);
	}

}
void readLineToInstance_response(const string &line, Instance *instance) {

	vector<string>vec;
	vec.clear();
	string sub = "";
	bool is_space = false;
	for (int i = 0; i < line.length(); i++) {
		if (line[i] == ' ' || line[i] == 9) {
			if (is_space) continue;
			vec.push_back(sub);
			sub = "";
			is_space = true;
			continue;
		}
		sub = sub + line[i];
		is_space = false;
	}
	vec.push_back(sub);
        string label = vec.at(0);
        if(label == "p") instance->m_label = "FAVOR";
        else if(label == "n") instance->m_label = "AGAINST";
        else if(label == "m") instance->m_label = "NONE";
        else{
            cout<<"read wrong label!!!"<<endl;
            abort();
        }
	vector<string>::iterator k = vec.begin();
        vec.erase(k);
        k = vec.begin();
        vec.erase(k);
	for (int i = 0; i < vec.size();i++) {
            string word = normalize_to_lowerwithdigit(vec[i]);
         //   if (word[0] == '@') continue;
	    instance->m_words.push_back(word);
	}

}
vector<Instance> readInstancesFromFile(const string &fullFileName) {
	vector<string> lines = readLines(fullFileName);
	vector<Instance> instances;

	using std::move;
	for (int i = 0; i < lines.size() - 3; i = i + 3) {
		Instance ins;
		readLineToInstance_post(lines.at(i), &ins);
		readLineToInstance_response(lines.at(i + 1), &ins);
		instances.push_back(move(ins));
	}

	return instances;
}

#endif

