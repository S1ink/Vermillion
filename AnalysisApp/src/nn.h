#pragma once

#include <memory>
#include <vector>
#include <array>
#include <functional>
#include <type_traits>
#include <random>

#include <eigen-3.4.0/Eigen>


using Scalar_t = float;
typedef Eigen::Matrix<Scalar_t, -1, -1, Eigen::RowMajor>	Matrix;
typedef Eigen::RowVectorX<Scalar_t>							VecR;
typedef Eigen::VectorX<Scalar_t>							VecC;

typedef std::vector<std::unique_ptr<VecR> >		MultiRow_Uq;
typedef std::vector<std::unique_ptr<Matrix> >	MultiMat_Uq;


template<typename scalar>
inline static scalar sigmoid(scalar x) { return (scalar)1 / (1 + exp(-x)); }
template<typename scalar>
inline static scalar sigmoid_d(scalar x) { return sigmoid(x) * (1 - sigmoid(x)); }
template<typename scalar>
inline static scalar hyperbolictan(scalar x) {
	if constexpr (std::is_floating_point<scalar>::value) { return tanhf(x); }
	return tanh(x);
}
template<typename scalar>
inline static scalar hyperbolictan_d(scalar x) { return 1 - (hyperbolictan(x) * hyperbolictan(x)); }
template<typename scalar>
inline static scalar relu(scalar x) { return x > 0 ? x : 0; }
template<typename scalar>
inline static scalar relu_d(scalar x) { return x > 0 ? (scalar)1 : (scalar)0; }


enum ActivationFunc {
	SIGMOID,
	TANH,
	RELU
};

template<typename scalar>
inline static scalar activation(scalar x, ActivationFunc f) {
	switch (f) {
		case SIGMOID: return sigmoid<scalar>(x);
		case TANH: return hyperbolictan<scalar>(x);
		case RELU: return relu<scalar>(x);
	}
}
template<typename scalar>
inline static std::function<scalar(scalar)> getFunc(ActivationFunc f) {
	switch (f) {
		case SIGMOID: return sigmoid<scalar>;
		case TANH: return hyperbolictan<scalar>;
		case RELU: return relu<scalar>;
	}
}
template<typename scalar>
inline static scalar activation_deriv(scalar x, ActivationFunc f) {
	switch (f) {
		case SIGMOID: return sigmoid_d<scalar>(x);
		case TANH: return hyperbolictan_d<scalar>(x);
		case RELU: return relu_d<scalar>(x);
	}
}
template<typename scalar>
inline static std::function<scalar(scalar)> getFuncDeriv(ActivationFunc f) {
	switch (f) {
		case SIGMOID: return sigmoid_d<scalar>;
		case TANH: return hyperbolictan_d<scalar>;
		case RELU: return relu_d<scalar>;
	}
}


template<typename t>
inline t randomRange(t l, t h) { return (rand() / (t)RAND_MAX) * (h - l) + l; }

template<typename scalar>
struct IOFunc_ {
	typedef Eigen::RowVectorX<scalar>	VecR;

	inline static constexpr scalar max_coeff{ 10 };

	IOFunc_() = default;
	inline IOFunc_(size_t i, size_t o) { this->gen(i, o); }

	void gen(size_t i, size_t o);
	bool execute(VecR& i, VecR& o) const;
	inline bool operator()(VecR& i, VecR& o) const { return this->execute(i, o); }
	void serializeFunc(std::ostream&) const;
	void serializeStructure(std::ostream&) const;

	inline size_t inputs() const { return this->in; }
	inline size_t outputs() const { return this->out; }

protected:
	inline bool checkSize(const VecR& i, const VecR& o) const {
		return (i.cols() == this->in && o.cols() == this->out) || this->in == 0 || this->out == 0;
	}

private:
	size_t in{0}, out{0};
	std::vector<std::array<size_t, 3> > insert_loc;		// { {Idx, Row, Col}, ... }
	mutable Eigen::RowVectorX<scalar> a;
	mutable Eigen::MatrixX<scalar> b;

};
typedef IOFunc_<Scalar_t>	IOFunc;


class NeuralNetwork {
public:
	typedef std::unique_ptr<VecR>		Layer_t;
	typedef std::unique_ptr<Matrix>		Weight_t;
	typedef std::vector<Layer_t>		Layers_t;
	typedef std::vector<Weight_t>		Weights_t;

	typedef std::vector<std::unique_ptr<VecR> >							IOList;
	typedef std::pair<IOList, IOList>									DataSet;
	typedef std::pair<std::unique_ptr<VecR>, std::unique_ptr<VecR> >	DataPoint;
	typedef std::vector<DataPoint>										DataSet_;


	NeuralNetwork(
		const std::vector<size_t>& topo,
		Scalar_t learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID
	);
	NeuralNetwork(
		std::vector<size_t>&& topo,
		Scalar_t learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID
	);
	NeuralNetwork(
		Weights_t&& weights,
		Scalar_t learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID
	);

	void regenerate();
	void regenerate(std::vector<size_t>&&);
	void regenerate(Weights_t&&);

	void propegateForward(const VecR& input) const;
	void propegateBackward(const VecR& output);
	void calcErrors(const VecR& output);
	void updateWeights();

	inline void train(const DataSet& d) { this->train(d.first, d.second); }
	void train(const IOList& data_in, const IOList& data_out);
	inline void train_verbose(const DataSet& d) { this->train_verbose(d.first, d.second); }
	void train_verbose(const IOList& data_in, const IOList& data_out);
	inline void train_graph(const DataSet& d, std::vector<float>& p) { this->train_graph(d.first, d.second, p); }
	void train_graph(const IOList& data_in, const IOList& data_out, std::vector<float>& progress);
	inline void train_instance(const DataPoint& d) { this->train_instance(*d.first, *d.second); }
	float train_instance(const VecR& in, const VecR& out);
	void inference(const VecR& in, VecR& out) const;

	void export_weights(std::ostream& out);
	static void parse_weights(std::istream& in, Weights_t& weights);

	void setActivationFunc(ActivationFunc f);
	inline void setLearningRate(Scalar_t lr) { this->learning_rate = lr; }
	inline const Scalar_t& getLearningRate() const { return this->learning_rate; }
	inline size_t inputs() const { return this->topology.front(); }
	inline size_t outputs() const { return this->topology.back(); }
	//size_t computeTotalSize() const;
	size_t computeHorizontalUnits() const;
	//size_t computeVerticalUnits() const;

	inline bool compatibleFunc(const IOFunc& f) const { return this->inputs() == f.inputs() && this->outputs() == f.outputs(); }
	inline bool compatibleDataSet(const DataSet& d) const { return this->inputs() == d.first.size() && this->outputs() == d.second.size(); }

	inline void genFunc(IOFunc& f) const { f.gen(this->inputs(), this->outputs()); }
	void genFuncData(DataSet& d, size_t s, IOFunc& f = IOFunc{}) const;
	void genData(DataSet& d, size_t s, const IOFunc& f) const;

	static void exportData(DataSet& d, std::ostream& o);
	static void importData(DataSet& d, std::istream& i);


protected:
	mutable Layers_t neurons_matx;
	Layers_t cache_neurons, deltas;
	Weights_t weights;
	std::vector<size_t> topology;
	Scalar_t learning_rate;
	std::function<Scalar_t(Scalar_t)>
		activation_func,
		activation_func_deriv;


};


//template<size_t... Args> struct select_first_size_t;
//template<size_t A, size_t ...Args> struct select_first_size_t<A, Args...> { static constexpr size_t value = A; };
//
//template<size_t A> struct tag{ static constexpr size_t value = A; };
//template<size_t... Args> struct select_last_size_t { static constexpr size_t value = decltype((tag<Args>{}, ...))::value; };
//
//
//template<size_t... topology>
//class NeuralNetwork_ {
//public:
//	static constexpr size_t
//		layers = sizeof...(topology),
//		inputs = select_first_size_t<topology...>::value,
//		outputs = select_last_size_t<topology...>::value;
//	static_assert(layers > 2, "Cannot construct NN with less than 2 layers");
//
//	typedef std::unique_ptr<VecR>				Layer_t;
//	typedef std::unique_ptr<Matrix>				Weight_t;
//	typedef std::array<Layer_t, layers>			Layers_t;
//	typedef std::array<Weight_t, layers - 1>	Weights_t;
//
//
//	inline NeuralNetwork_(Scalar_t lrate = 0.005) : learning_rate(lrate) {
//		this->init();
//	}
//	NeuralNetwork_(Weights_t&& weights);
//	NeuralNetwork_(MultiMat_Uq&& weights);
//
//	void propegateForward(const VecR& input) const;
//	void propegateBackward(VecR& output);
//	void calcErrors(VecR& output);
//	void updateWeights();
//	void train(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
//	void train_verbose(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
//	void train_graph(MultiRow_Uq& data_in, MultiRow_Uq& data_out, std::vector<float>& progress);
//	float train_instance(VecR& in, VecR& out);
//	void inference(const VecR& in, VecR& out) const;
//
//
//protected:
//	template<size_t l = 0, size_t... ls>
//	void init_(size_t i = 0, size_t last = 0) {
//		if constexpr (i == layers) {
//			return;
//		}
//		static_assert(l > 0, "Layer must have at least 1 neuron");
//
//		if constexpr (i == layers - 1) {
//			this->neurons_matx[i] = std::make_unique<VecR>(l);
//			this->cache_neurons[i] = std::make_unique<VecR>(l);
//			this->deltas[i] = std::make_unique<VecR>(l);
//		} else {
//			this->neurons_matx[i] = std::make_unique<VecR>(l + 1);
//			this->cache_neurons[i] = std::make_unique<VecR>(l + 1);
//			this->deltas[i] = std::make_unique<VecR>(l + 1);
//
//			this->neurons_matx[i]->coeffRef(l) = 1.0;
//			this->cache_neurons[i]->coeffRef(l) = 1.0;
//		}
//
//		if constexpr (i > 0) {
//			if constexpr (i != layers - 1) {
//				this->weights[i] = std::make_unique<Matrix>(last + 1, l + 1);
//				this->weights[i]->setRandom();
//				this->weights[i]->col(l).setZero();
//				this->weights[i]->coeffRef(last + 1, l) = 1.0;
//			} else {
//				this->weights[i] = std::make_unique<Matrix>(last + 1, l);
//				this->weights[i]->setRandom();
//			}
//		}
//
//		init_<ls...>(i + 1, l);
//	}
//
//	inline void init() {
//		this->init_< topology... >();
//	}
//
//	mutable Layers_t neurons_matx;
//	Layers_t cache_neurons, deltas;
//	Weights_t weights;
//	Scalar_t learning_rate;
//
//
//};



template<typename s>
void IOFunc_<s>::gen(size_t i, size_t o) {
	// or generate random between i and [?] to use like size i >>
	this->a.resize(i);
	this->a.setRandom();
	this->b.resize(i, o);
	this->b.setRandom();
	for (auto& z : this->a) {
		z = (int)(z * this->max_coeff);
	}
	for (auto& z : this->b.reshaped()) {
		z = (int)(z * this->max_coeff);
	}

	this->insert_loc.resize(i);
	size_t x = rand();
	for (size_t k = 0; k < i; k++) {
		this->insert_loc[k][0] = x++ % 2;
		int r, c;
		if (this->insert_loc[k][0] == 0) {
			r = this->a.rows();
			c = this->a.cols();
		} else {
			r = this->b.rows();
			c = this->b.cols();
		}
		this->insert_loc[k][1] = randomRange<float>(0, r);
		this->insert_loc[k][2] = randomRange<float>(0, c);
	}

	this->in = i;
	this->out = o;

}
template<typename s>
bool IOFunc_<s>::execute(VecR& i, VecR& o) const {
	if (!this->checkSize(i, o)) {
		return false;
	}
	for (size_t k = 0; k < this->in; k++) {
		if (this->insert_loc[k][0] == 0) {
			this->a(this->insert_loc[k][2]) = i[k];
		} else {
			this->b(this->insert_loc[k][0], this->insert_loc[k][2]) = i[k];
		}
	}
	o = this->a * this->b;
	return true;
}
template<typename s>
void IOFunc_<s>::serializeFunc(std::ostream& o) const {
	for (size_t i = 0; i < this->out; i++) {	// for each output ~~ number of cols in b
		o << i + 1 << ". ";
		for (size_t c = 0; c < this->a.cols(); c++) {	// for each part ~~ number of cols in a ~~ number of rows in b
			s e1 = this->a[c];
			s e2 = this->b(c, i);
			char ec1{ '\0' }, ec2{ '\0' };
			for (size_t l = 0; l < this->in; l++) {
				if (this->insert_loc[l][0] == 0 && this->insert_loc[l][2] == c) {
					ec1 = 'a' + l;
				} else if (this->insert_loc[l][1] == c && this->insert_loc[l][2] == i) {
					ec2 = 'a' + l;
				}
			}
			o << "(";
			if (ec1 == '\0') { o << e1; }
			else { o << ec1; }
			o << " * ";
			if (ec2 == '\0') { o << e2; }
			else { o << ec2; }
			o << ") " << (c != this->a.cols() - 1 ? "+ " : "\n");
		}
		o.flush();
	}
	o << std::endl;
}
template<typename s>
void IOFunc_<s>::serializeStructure(std::ostream& o) const {
	o << "[";
	for (size_t i = 0; i < this->in; i++) {
		for (size_t k = 0; k < this->insert_loc.size(); k++) {
			if (this->insert_loc[k][0] == 0 && this->insert_loc[k][2] == i) {
				o << (char)('a' + k);
				goto cont;
			}
		}
		o << this->a[i];
	cont:
		if (i != this->in - 1) {
			o << ", ";
		}
	}
	o << "]\n[\n";
	for (size_t r = 0; r < this->in; r++) {
		o << '\t';
		for (size_t c = 0; c < this->out; c++) {
			for (size_t k = 0; k < this->insert_loc.size(); k++) {
				if (this->insert_loc[k][0] == 1 && this->insert_loc[k][1] == r && this->insert_loc[k][2] == c) {
					o << (char)('a' + k);
					goto cont2;
				}
			}
			o << this->b(r, c);
		cont2:
			if (c != this->out - 1 || r != this->in - 1) {
				o << ", ";
			}
		}
		o << '\n';
	}
	o << "]\n\n";
	o.flush();
}