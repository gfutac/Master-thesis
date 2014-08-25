#ifndef LOADBAR_H
#define LOADBAR_H

#include <iostream>

using namespace std;

inline void loadBar(int len, double percent) {
	cout << "\b\r"; 
	string progress, l;
	stringstream line;

	if (static_cast<int>(100 * percent) % 10 == 0) {
		for (int i = 0; i < len; ++i) {
			if (i < static_cast<int>(len * percent)) {
				progress += "=";
			} else {
				progress += " ";
			}
		}

		line << "\t[" << progress << "] " << static_cast<int>(100 * percent) << "%";
		cout << line.str();	
	}
	
	if (static_cast<int>(100 * percent) == 100){
		stringstream d;
		for (int i = 0; i < line.str().length(); ++i) d << "\b \b";
		cout << d.str();
	}

	flush(cout);
	cout << "\b\r"; 
}

#endif LOADBAR_H