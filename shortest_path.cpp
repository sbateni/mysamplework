



//============================================================================
// Name        :shortest_path.cpp
// Author      : Ehsan Bateni
// Version     :
// Copyright   : Your copyright notice
// Description : Computes the shortest path between two nodes in a graph using Breadth-first search
//============================================================================

#include "defs.hpp"

class vertex {

	int value;
	int parent;
	int d;
	std::string color;
public:
	vertex() {
		color = "white";
		d = 0;
	}
	friend class graph;
	~vertex() {
	}

};

class graph {

public:
	int size;
	List *Adj = new List[size];
	friend class vertex;
	graph(int s) :
			size(s) {
	}

	void addedge(int v1, int v2) {
		if (!Adj[v1].IsinList(v2))
			Adj[v1].Append(v2);
		if (!Adj[v2].IsinList(v1))
			Adj[v2].Append(v1);
	}

	List* shortestpath(int s, int v) {
		List *Q = new List;
		List *path = new List;
		vertex *V = new vertex[size];
		Node *a_n;
		int u = 0;
		int a = 0;
		Q->Append(s);
		V[s].value = s;
		V[s].color = "black";
		V[s].d = 0;

		do {

			u = Q->get_head()->Data();
			Q->Delete(u);
			a_n = Adj[u].get_head();

			do {
				a = a_n->Data();
				V[a].value = a;
				a_n = a_n->Next();

				if (V[a].color == "white") {

					V[a].color = "gray";
					V[a].parent = u;
					V[a].d = V[u].d + 1;

					Q->Append(a);
					//	Q.Print();
				}

				V[u].color = "black";

			} while (a_n);

		} while (u != v && Q->get_head());

		int p = u;
		if (u != v) {
			return path;
		}

		for (int i = 0; i <= V[u].d; i++) {

			path->Insert(p);
			p = V[p].parent;

		}

		delete[] V;
		delete Q;
		return path;
	}

	~graph() {
		delete[] Adj;

	}

};

int main() {

	std::string line;

	char cmd;
	int V, first, second;
	graph *G;
	Node *tmp;
	List *p;
	int empty_edge_flag = 0;
	List *argE = new List;

	char state = 'V';
	while (true) {

		std::getline(std::cin, line);

		if (std::cin.eof()) {
			break;
		}

		List *arg = new List;

		std::string err_msg;
		if (parse_line(line, cmd, arg, err_msg, state)) {
			switch (cmd) {

			case 'V':

				V = arg->get_head()->Data();
				delete arg;
				G = new graph(V);
				state = 'E';
				break;

			case 'E':
				empty_edge_flag = 0;
				delete argE;
				argE = new List;
				tmp = arg->get_head();
				if (!tmp) {
					empty_edge_flag = 1;
					state = 'V';
					break;
				}
				do {

					first = tmp->Data();
					argE->Append(first);
					tmp = tmp->Next();
					second = tmp->Data();
					argE->Append(second);
					tmp = tmp->Next();
					if (first >= V || second >= V) {
						std::cerr << "Error: "
								<< "indice exceeds vertice numbers"
								<< std::endl;
						delete arg;
						break;
					}

					if (first != second)
						G->addedge(first, second);
					arg->Delete(first);
					arg->Delete(second);
					state = 's';
				} while (tmp);

				break;

			case 's':

				first = arg->get_head()->Data();
				second = arg->get_head()->Next()->Data();
				if (first >= V || second >= V || empty_edge_flag) {
					std::cerr << "Error: " << "indice exceeds vertice numbers"
							<< std::endl;
					delete arg;
					break;
				}
				if (first == second) {
					std::cout << first << std::endl;
					delete arg;
					break;
				}
				if (!argE->IsinList(first) || !argE->IsinList(second)) {
					std::cerr << "Error: "
							<< "there is no path between vertices" << std::endl;
					delete arg;
					break;
				}

				p = G->shortestpath(first, second);

				if (!p->get_head()) {
					std::cerr << "Error: "
							<< "there is no path between vertices" << std::endl;
					delete arg;
					break;
				}
				p->Print();
				delete p;
				delete arg;
				state = 's';
				break;

			}

		} else {
			std::cerr << "Error: " << err_msg << std::endl;
		}
	}

	return 0;
}



