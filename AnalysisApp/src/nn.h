#pragma once

#include <memory>
#include <vector>
#include <array>
#include <functional>
#include <type_traits>

#include <eigen-3.4.0/Eigen>


using Scalar_t = float;
typedef Eigen::Matrix<Scalar_t, -1, -1>		Matrix;
typedef Eigen::Matrix<Scalar_t, 1, -1>		VecR;
typedef Eigen::Matrix<Scalar_t, -1, 1>		VecC;

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


class NeuralNetwork {
public:
	typedef std::unique_ptr<VecR>		Layer_t;
	typedef std::unique_ptr<Matrix>		Weight_t;
	typedef std::vector<Layer_t>		Layers_t;
	typedef std::vector<Weight_t>		Weights_t;

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
	void propegateBackward(VecR& output);
	void calcErrors(VecR& output);
	void updateWeights();
	void train(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
	void train_verbose(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
	void train_graph(MultiRow_Uq& data_in, MultiRow_Uq& data_out, std::vector<float>& progress);
	float train_instance(VecR& in, VecR& out);
	void inference(const VecR& in, VecR& out) const;

	void export_weights(std::ostream& out);
	static void parse_weights(std::istream& in, Weights_t& weights);

	void setActivationFunc(ActivationFunc f);
	inline void setLearningRate(Scalar_t lr) { this->learning_rate = lr; }
	inline const Scalar_t& getLearningRate() const { return this->learning_rate; }


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