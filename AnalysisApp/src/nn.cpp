#include "nn.h"

#include <iostream>


void NeuralNetwork::regenerate() {
	this->neurons_matx.clear();
	this->cache_matx.clear();
	this->errors.clear();
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

		this->cache_matx.emplace_back(	// rvecs of the size of current neuron layer amount (?)
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// changed from 'neurons_matx.size()'
		this->errors.emplace_back(			// ^^^
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// '''
		this->cache_matx.back()->setZero();
		this->errors.back()->setZero();
		// topo{2, 3, 1} -> cache/delta{1, 2, 3} [x] -> cache/delta{3, 4, 1}

		if (i != this->topology.size() - 1) {	// not last idx
			this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;	// last coeff(bias) in layer rvec = 1.0
			this->cache_matx.back()->coeffRef(this->topology.at(i)) = 1.0;	// accessing out of bounds?
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
						this->topology.at(i)			// this layer size (no bias in output layer)
					));
				this->weights.back()->setRandom();	// assign random starting values
				this->weights.back()->row(this->weights.back()->rows() - 1).setZero();
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
		this->cache_matx.clear();
		this->errors.clear();

		this->topology.push_back( this->weights.at(0)->rows() - 1U );
		for (size_t i = 0; i < this->weights.size(); i++) {	// generate topo using the inverse of weight generation above
			this->topology.push_back(
				this->weights.at(i)->cols() - 1U );
		}
		this->topology.back()++;
		for (size_t i = 0; i < this->topology.size(); i++) {	// same as above
			if (i == topology.size() - 1) {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i))	);
			} else {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i) + 1)	);
			}

			this->cache_matx.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);
			this->errors.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);
			this->cache_matx.back()->setZero();
			this->errors.back()->setZero();

			if (i != this->topology.size() - 1) {
				this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;
				this->cache_matx.back()->coeffRef(this->topology.at(i)) = 1.0;
			}

		}
	}
}

void NeuralNetwork::remix() {
	for (size_t i = 0; i < this->weights.size(); i++) {
		this->weights[i]->setRandom();
		if (i != this->weights.size() - 1) {
			this->weights[i]->col(this->weights[i]->cols() - 1).setZero();
			this->weights[i]->coeffRef(
				this->weights[i]->rows() - 1,
				this->weights[i]->cols() - 1
			) = 1.0;
		} else {
			this->weights.back()->row(this->weights.back()->rows() - 1).setZero();
		}
	}
}


NeuralNetwork::NeuralNetwork(
	const std::vector<size_t>& topo, float lr, ActivationFunc f, Regularization r, float rr
) : NeuralNetwork(std::move(std::vector<size_t>(topo)), lr, f, r, rr) {}
NeuralNetwork::NeuralNetwork(
	std::vector<size_t>&& topo, float lr, ActivationFunc f, Regularization r, float rr
) : 
	topology(std::move(topo)), learning_rate(lr), reg_rate(rr)
{
	this->setActivationFunc(f);
	this->setRegularization(r);
	this->regenerate();
}
NeuralNetwork::NeuralNetwork(
	Weights_t&& weights, float lr, ActivationFunc f, Regularization r, float rr
) :
	learning_rate(lr), reg_rate(rr)
{
	this->setActivationFunc(f);
	this->setRegularization(r);
	this->regenerate(std::move(weights));
}


void NeuralNetwork::propegateForward(const VecR& input) const {
	this->neurons_matx.front()->block(0, 0, 1, this->neurons_matx.front()->cols() - 1) = input;	// set the first neuron layer to input
	for (size_t i = 1; i < this->topology.size(); i++) {	// iterate through remaining layers
		if (i != this->topology.size() - 1) {
			*this->cache_matx[i] = (*this->neurons_matx[i - 1]) * (*this->weights[i - 1]);	// fill next neuron layer with previous * weights(connecting the 2 layers)
			this->neurons_matx[i]->block(0, 0, 1, this->neurons_matx[i]->cols() - 1) =
				this->cache_matx[i]->block(0, 0, 1, this->cache_matx[i]->cols() - 1).unaryExpr(this->activation_func);	// apply activation func to all neurons in layer
		} else {
			*this->neurons_matx[i] = (*this->neurons_matx[i - 1]) * (*this->weights[i - 1]);
		}
	}
}
void NeuralNetwork::propegateBackward(const VecR& output) {
	this->calcErrors(output);
	this->updateWeights();
}
void NeuralNetwork::calcErrors(const VecR& output) {
	*this->errors.back() =
		(output - *this->neurons_matx.back())/*.array() *
		this->cache_matx.back()->unaryExpr(this->activation_func_deriv).array()*/;
	for (size_t i = this->topology.size() - 2; i > 0; i--) {
		(*this->errors[i]) =
			((*this->errors[i + 1]) * (this->weights[i]->transpose())).array() *
			this->cache_matx[i]->unaryExpr(this->activation_func_deriv).array();	// store error for each layer by working backwards
		this->errors[i]->coeffRef(this->errors[i]->size() - 1) = 0;
		if (!this->errors[i]->allFinite()) {
			std::cout << "CalcErrors failure: NaN detected. Aborting." << std::endl;
			this->dump(std::cout);
			abort();
		}
	}
}
void NeuralNetwork::updateWeights() {
	size_t i = 0;
	for (; i < this->topology.size() - 1; i++) {
		if (this->reg_f == L2) {
			this->weights[i]->
				block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1)
				*= (1.f - this->reg_rate * this->learning_rate);
		}
		else if (this->regularization_func_deriv) {
			this->weights[i]->
				block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1)
				-= (this->weights[i]->
					block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1).unaryExpr(
						this->regularization_func_deriv
					) * this->reg_rate * this->learning_rate);
		}
		*this->weights[i] += (
			this->learning_rate * (this->neurons_matx[i]->transpose() * *this->errors[i + 1])
		);
		if (!this->weights[i]->allFinite()) {
			std::cout << "UpdateWeights failure: NaN detected. Aborting." << std::endl;
			this->dump(std::cout);
			abort();
		}
	}
	this->weights[i - 1]->row(this->weights[i - 1]->rows() - 1).setZero();

////		if (i != this->topology.size() - 2) {
//			for (size_t c = 0; c < this->weights[i]->cols() - (i != this->topology.size() - 2); c++) {	// skip last col exept on last matx
//				for (size_t r = 0; r < this->weights[i]->rows() - 1; r++) {	// skip last row on last matx
//					this->weights[i]->coeffRef(r, c) +=
//						this->learning_rate *
//						this->errors[i + 1]->coeff(c) *
//						//this->activation_func_deriv(this->cache_matx[i + 1]->coeffRef(c)) *
//						this->neurons_matx[i]->coeff(r);	// for biases delete this part, just add lr * err
//						// also subtract regularization value
//				}
//				if (i != this->topology.size() - 2) {
//					this->weights[i]->coeffRef(this->weights[i]->rows() - 1, c) +=
//						this->learning_rate * this->errors[i + 1]->coeff(c);
//				}
//			}
//			
////		}
//		//else {
//		//	for (size_t c = 0; c < this->weights[i]->cols(); c++) {
//		//		for (size_t r = 0; r < this->weights[i]->rows(); r++) {
//		//			this->weights[i]->coeffRef(r, c) +=
//		//				this->learning_rate *
//		//				this->errors[i + 1]->coeff(c) *
//		//				//this->activation_func_deriv(this->cache_matx[i + 1]->coeffRef(c)) *
//		//				this->neurons_matx[i]->coeff(r);
//		//		}
//		//	}
//		//}
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
			(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size()
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
			(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size()));
	}
}
float NeuralNetwork::train_instance(const VecR& in, const VecR& out) {
	this->propegateForward(in);
	propegateBackward(out);
	return std::sqrt(
		(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size());
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
void NeuralNetwork::setRegularization(Regularization f) {
	this->reg_f = f;
	this->regularization_func = getRegFunc<Scalar_t>(f);
	this->regularization_func_deriv = getRegFuncDeriv<Scalar_t>(f);
}
//size_t NeuralNetwork::computeTotalSize() const {
//	size_t ret{ 0 };
//	for (size_t i = 0; i < this->weights.size(); i++) {
//		ret += this->weights[i]->size();
//	}
//	return ret;
//}
size_t NeuralNetwork::computeHorizontalUnits() const {
	size_t ret{ this->topology.size() };
	for (size_t i = 0; i < this->weights.size(); i++) {
		ret += this->weights[i]->cols();
	}
	return ret;
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
void NeuralNetwork::exportData_strictCSV(DataSet& d, std::ostream& oi, std::ostream& oo) {
	for (size_t s = 0; s < d.first.size(); s++) {
		VecR& ins = *d.first[s];
		VecR& outs = *d.second[s];
		for (size_t i = 0; i < ins.size(); i++) {
			oi << ins[i];
			oi << (i != ins.size() - 1 ? ", " : "\n");
		}
		oi.flush();
		for (size_t i = 0; i < outs.size(); i++) {
			oo << outs[i];
			oo << (i != outs.size() - 1 ? ", " : "\n");
		}
		oo.flush();
	}
}
void NeuralNetwork::importData(DataSet& d, std::istream& i) {
	std::string sect;
	std::vector<Scalar_t> buff;
	size_t pos, split;
	while (std::getline(i, sect, '\n')) {
		buff.clear();
		pos = split = 0;
		std::istringstream stream(sect);
		split = sect.find(':', 0);
		if (split != std::string::npos) {
			for (;;) {
				buff.emplace_back();
				stream >> buff.back();
				pos = sect.find(',', stream.tellg());
				if (stream.tellg() == (std::istringstream::pos_type)(-1)) {
					d.second.emplace_back(std::make_unique<VecR>(
						Eigen::Map<VecR>(buff.data(), buff.size())
					));
					break;
				} else if (pos - stream.tellg() < 3) {
					stream.ignore(2, ',');
				} else if (pos > split && (split - stream.tellg()) < 3) {
					d.first.emplace_back(std::make_unique<VecR>(
						Eigen::Map<VecR>(buff.data(), buff.size())
					));
					buff.clear();
					stream.ignore(2, ':');
				} else {
					// ???
				}
			}
		} else {
			// no separator
		}
		if (d.first.size() > 1 && d.first.back()->size() != d.first[d.first.size() - 2]->size()) {
			// error
		}
		if (d.second.size() > 1 && d.second.back()->size() != d.second[d.second.size() - 2]->size()) {
			// error
		}
		
	}
}
void NeuralNetwork::importData_strictCSV(DataSet& d, std::istream& ii, std::istream& io) {

}

void NeuralNetwork::dump(std::ostream& out) {
	out << "Weights:\n";
	for (size_t i = 0; i < this->weights.size(); i++) {
		out << '[' <<  *this->weights[i] << "]\n";
	}
	out.flush();
	out << "\nNeurons:\n";
	for (size_t i = 0; i < this->neurons_matx.size(); i++) {
		out << '[' << *this->neurons_matx[i] << "]\n";
	}
	out.flush();
	out << "\nCaches:\n";
	for (size_t i = 0; i < this->neurons_matx.size(); i++) {
		out <<  '[' << *this->cache_matx[i] << "]\n";
	}
	out.flush();
	out << "\nDeltas:\n";
	for (size_t i = 0; i < this->errors.size(); i++) {
		out << '[' << *this->errors[i] << "]\n";
	}
	out << "\n\n";
	out.flush();
}