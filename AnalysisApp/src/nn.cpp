#include "nn.h"

#include <iostream>


void NeuralNetwork::regenerate() {
	this->neurons_matx.clear();
	this->cache_neurons.clear();
	this->deltas.clear();
	this->weights.clear();

	for (size_t i = 0; i < this->topology.size(); i++) {
		if (i == topology.size() - 1) {		// last(output) layer
			this->neurons_matx.emplace_back(	// add layer of size of last layer in topo
				std::make_unique<VecR>(this->topology.at(i))	);
		} else {
			this->neurons_matx.emplace_back(	// add layer of size of match topo + 1 for bias
				std::make_unique<VecR>(this->topology.at(i) + 1)	);
		}
		// topo{2, 3, 1} -> neurons{3, 4, 1}

		this->cache_neurons.emplace_back(	// rvecs of the size of current neuron layer amount (?)
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// changed from 'neurons_matx.size()'
		this->deltas.emplace_back(			// ^^^
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// '''
		// topo{2, 3, 1} -> cache/delta{1, 2, 3} [x] -> cache/delta{3, 4, 1}

		if (i != this->topology.size() - 1) {	// not last idx
			this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;	// last coeff(bias) in layer rvec = 1.0
			this->cache_neurons.back()->coeffRef(this->topology.at(i)) = 1.0;	// accessing out of bounds?
		}
		/*
		* topo{2, 3, 1} ->
		* neurons
		* { {n, n, 1},
		*	{n, n, n, 1},
		*	{n} }
		* cache
		* { {0, 0, 1},
		*	{0, 0, 0, 1},
		*	{0} }
		* delta
		* { {0, 0, 0},
		*	{0, 0, 0, 0},
		*	{0} }
		*/

		if (i > 0) {	// not first idx
			if (i != topology.size() - 1) {	// and not last idx
				this->weights.emplace_back(
					std::make_unique<Matrix>(	// resizable matrix
						this->topology.at(i - 1) + 1,	// last layer size plus 1
						this->topology.at(i) + 1		// this layer size plus 1 for bias
					));
				this->weights.back()->setRandom();	// randomize starting weight
				this->weights.back()->col(this->topology.at(i)).setZero();	// set last col to all zeros
				this->weights.back()->coeffRef(	// set outermost coeff to 1 (highest of ^^^)
					this->topology.at(i - 1),
					this->topology.at(i)
				) = 1.0;
			} else {						// last idx
				this->weights.emplace_back(
					std::make_unique<Matrix>(
						this->topology.at(i - 1) + 1,	// last layer size plus 1
						this->topology.at(i)			// this layer size (no bias)
					));
				this->weights.back()->setRandom();	// assign random starting values
			}
		}
		/*
		* topo{2, 3, 1} ->
		* weights(s)
		* { {3, 4},
		*	{4, 1} }
		* ~~~>>>
		* weights
		* { {
		*	{r, r, r, 0},
		*	{r, r, r, 0},
		*	{r, r, r, 1}
		* }, {
		*	{r},
		*	{r},
		*	{r},
		*	{r}
		* } }
		*/

	}
}
void NeuralNetwork::regenerate(std::vector<size_t>&& topo) {
	if (topo.size() > 1) {
		this->topology = std::move(topo);
		this->regenerate();
	}
}
void NeuralNetwork::regenerate(Weights_t&& weights) {
	if (!weights.empty()) {
		this->weights = std::move(weights);
		this->topology.clear();
		this->neurons_matx.clear();
		this->cache_neurons.clear();
		this->deltas.clear();

		this->topology.push_back( this->weights.at(0)->rows() - 1U );
		for (size_t i = 0; i < this->weights.size(); i++) {	// generate topo using the inverse of weight generation above
			this->topology.push_back(
				this->weights.at(i)->cols() - 1U );
		}
		for (size_t i = 0; i < this->topology.size(); i++) {	// same as above
			if (i == topology.size() - 1) {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i))	);
			} else {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i) + 1)	);
			}

			this->cache_neurons.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);
			this->deltas.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);

			if (i != this->topology.size() - 1) {
				this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;
				this->cache_neurons.back()->coeffRef(this->topology.at(i)) = 1.0;
			}

		}
	}
}


NeuralNetwork::NeuralNetwork(
	const std::vector<size_t>& topo, Scalar_t lr, ActivationFunc f
) : NeuralNetwork(std::move(std::vector<size_t>(topo)), lr, f) {}
NeuralNetwork::NeuralNetwork(
	std::vector<size_t>&& topo, Scalar_t lr, ActivationFunc f
) : 
	topology(std::move(topo)), learning_rate(lr), activation_func(getFunc<Scalar_t>(f)), activation_func_deriv(getFuncDeriv<Scalar_t>(f))
{
	this->regenerate();
}
NeuralNetwork::NeuralNetwork(
	Weights_t&& weights, Scalar_t lr, ActivationFunc f
) :
	learning_rate(lr), activation_func(getFunc<Scalar_t>(f)), activation_func_deriv(getFuncDeriv<Scalar_t>(f))
{
	this->regenerate(std::move(weights));
}


void NeuralNetwork::propegateForward(const VecR& input) const {
	this->neurons_matx.front()->block(
		0, 0, 1, this->neurons_matx.front()->size() - 1) = input;	// set the first neuron layer to input

	for (size_t i = 1; i < this->topology.size(); i++) {	// iterate through remaining layers
		(*this->neurons_matx.at(i)) =
			(*this->neurons_matx.at(i - 1)) * (*this->weights.at(i - 1));	// fill next neuron layer with previous * weights(connecting the 2 layers)
		this->neurons_matx.at(i)->block(
			0, 0, 1, this->topology.at(i)
		).unaryExpr(this->activation_func);	// apply activation func to all neurons in layer
	}
}
void NeuralNetwork::propegateBackward(const VecR& output) {
	this->calcErrors(output);
	this->updateWeights();
}
void NeuralNetwork::calcErrors(const VecR& output) {
	(*this->deltas.back()) = output - (*this->neurons_matx.back());	// difference between target and calculated output layer
	for (size_t i = this->topology.size() - 2; i > 0; i--) {
		(*this->deltas.at(i)) = (*this->deltas.at(i + 1)) * (this->weights.at(i)->transpose());	// store error for each layer by working backwards
	}
}
void NeuralNetwork::updateWeights() {
	for (size_t i = 0; i < this->topology.size() - 1; i++) {
		if (i != this->topology.size() - 1) {
			for (size_t c = 0; c < this->weights.at(i)->cols() - 1; c++) {
				for (size_t r = 0; r < this->weights.at(i)->rows(); r++) {
					this->weights.at(i)->coeffRef(r, c) +=
						this->learning_rate *
						this->deltas.at(i + 1)->coeffRef(c) *
						this->activation_func_deriv(this->cache_neurons.at(i + 1)->coeffRef(c)) *
						this->neurons_matx.at(i)->coeffRef(r);
				}
			}
		}
		else {
			for (size_t c = 0; c < this->weights.at(i)->cols(); c++) {
				for (size_t r = 0; r < this->weights.at(i)->rows(); r++) {
					this->weights.at(i)->coeffRef(r, c) +=
						this->learning_rate *
						this->deltas.at(i + 1)->coeffRef(c) *
						this->activation_func_deriv(this->cache_neurons.at(i + 1)->coeffRef(c)) *
						this->neurons_matx.at(i)->coeffRef(r);
				}
			}
		}
	}
}
void NeuralNetwork::train(const IOList& data_in, const IOList& data_out) {
	for (size_t i = 0; i < data_in.size(); i++) {
		this->propegateForward(*data_in.at(i));
		propegateBackward(*data_out.at(i));
	}
}
void NeuralNetwork::train_verbose(const IOList& data_in, const IOList& data_out) {
	for (size_t i = 0; i < data_in.size(); i++) {
		std::cout << "Input to neural network is: " << *data_in.at(i) << '\n';
		this->propegateForward(*data_in.at(i));
		std::cout << "Expected output is: " << *data_out.at(i) <<
			"\nOutput produced is: " << *this->neurons_matx.back() << '\n';
		propegateBackward(*data_out.at(i));
		std::cout << "Itr: " << i << " - MSE: " << std::sqrt(
			(*this->deltas.back()).dot(*this->deltas.back()) / this->deltas.back()->size()
		) << '\n' << std::endl;
	}
	/*std::cout << "Weights:\n";
	for (size_t i = 0; i < this->weights.size(); i++) {
		std::cout << "{\n" << *this->weights.at(i) << "\n}\n";
	}*/
}
void NeuralNetwork::train_graph(const IOList& data_in, const IOList& data_out, std::vector<float>& progress) {
	//progress.clear();
	for (size_t i = 0; i < data_in.size(); i++) {
		this->propegateForward(*data_in.at(i));
		this->propegateBackward(*data_out.at(i));
		progress.push_back(std::sqrt(
			(*this->deltas.back()).dot(*this->deltas.back()) / this->deltas.back()->size()));
	}
}
float NeuralNetwork::train_instance(const VecR& in, const VecR& out) {
	this->propegateForward(in);
	propegateBackward(out);
	return std::sqrt(
		(*this->deltas.back()).dot(*this->deltas.back()) / this->deltas.back()->size());
}
void NeuralNetwork::inference(const VecR& in, VecR& out) const {
	this->propegateForward(in);
	out = *this->neurons_matx.back();
}


void NeuralNetwork::export_weights(std::ostream& out) {
	size_t wm = this->weights.size(), rm, cm;
	for (size_t w = 0; w < wm; w++) {
		out << "{\n";
		rm = this->weights[w]->rows();
		cm = this->weights[w]->cols();
		for (size_t r = 0; r < rm; r++) {
			out << "\t[ ";
			for (size_t c = 0; c < cm; c++) {
				out << this->weights[w]->coeff(r, c) << ' ';	// or '\t'
			}
			out << (r == rm - 1 ? "]\n" : "],\n");
		}
		out << (w == wm - 1 ? "}\n" : "},\n");
	}
}
void NeuralNetwork::parse_weights(std::istream& in, Weights_t& weights) {
	std::string line;
	std::vector<std::vector<Scalar_t> > buff;
	//size_t r = 0, c = 0;
	weights.clear();
	while (std::getline(in, line, '\n')) {
		if (line.length() > 0 && line.at(0) == '{') {
			// start new weights block
			buff.clear();
			while (std::getline(in, line, '\n') && line.at(0) != '}') {
				buff.emplace_back();
				std::istringstream str(line);
				str.ignore(2, '[');
				while (str.ignore(1) && str.peek() != ']') {
					buff.back().emplace_back();
					str >> buff.back().back();
					//std::cout << buff.back().back() << std::endl;
				}
			}
			weights.emplace_back(
				std::make_unique<Matrix>(
					buff.size(),
					buff[0].size()
					)
			);
			for (size_t r = 0; r < buff.size(); r++) {
				for (size_t c = 0; c < buff[r].size(); c++) {
					weights.back()->coeffRef(r, c) = buff[r][c];
					//std::cout << buff[r][c] << std::endl;
				}
			}
			if (line.length() == 1 && line.at(0) == '}') {
				return;
			}
		}
	}
}
void NeuralNetwork::setActivationFunc(ActivationFunc f) {
	this->activation_func = getFunc<Scalar_t>(f);
	this->activation_func_deriv = getFuncDeriv<Scalar_t>(f);
}


void NeuralNetwork::genFuncData(DataSet& d, size_t s, IOFunc& f) const {
	this->genFunc(f);
	d.first.clear();
	d.first.reserve(s);
	d.second.clear();
	d.second.reserve(s);
	size_t in = this->inputs(), out = this->outputs();
	for (size_t i = 0; i < s; i++) {
		d.first.emplace_back(std::make_unique<VecR>(in));
		d.second.emplace_back(std::make_unique<VecR>(out));
		d.first.back()->setRandom();
		if (!f(*d.first.back(), *d.second.back())) {
			std::cout << "Error generating dataset: idx[" << i << "]\n";
		}
	}
}
void NeuralNetwork::genData(DataSet& d, size_t s, const IOFunc& f) const {
	if (this->compatibleFunc(f)) {
		d.first.clear();
		d.first.reserve(s);
		d.second.clear();
		d.second.reserve(s);
		size_t in = this->inputs(), out = this->outputs();
		for (size_t i = 0; i < s; i++) {
			d.first.emplace_back(std::make_unique<VecR>(in));
			d.second.emplace_back(std::make_unique<VecR>(out));
			d.first.back()->setRandom();
			if (!f(*d.first.back(), *d.second.back())) {
				std::cout << "Error generating dataset: idx[" << i << "]\n";
			}
		}
	} else {
		std::cout << "Incompatible function IO size for generating dataset\n";
	}
}

void NeuralNetwork::exportData(DataSet& d, std::ostream& o) {
	for (size_t s = 0; s < d.first.size(); s++) {
		VecR& ins = *d.first[s];
		VecR& outs = *d.second[s];
		for (size_t i = 0; i < ins.size(); i++) {
			o << ins[i];
			if (i != ins.size() - 1) { o << ", "; }
		}
		o << " : ";
		for (size_t i = 0; i < outs.size(); i++) {
			o << outs[i];
			if (i != outs.size() - 1) { o << ", "; }
		}
		o << "\n";
	}
	o.flush();
}
void NeuralNetwork::importData(DataSet& d, std::istream& i) {

}