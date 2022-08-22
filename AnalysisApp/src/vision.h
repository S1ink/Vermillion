#pragma once

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <thread>
#include <initializer_list>

#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"
#include "imgui.h"

#include "Walnut/Application.h"
#include "Walnut/Image.h"

#include "nn.h"


struct Vec2f {
	union {
		struct {
			union { float x, width; };
			union { float y, height; };
		};
		ImVec2 imvec;
		cv::Size2f cvsize;
		float arr[2];
	};
};
struct Circle {
	union {
		struct {
			union {
				struct {
					float x, y;
				};
				cv::Point2f cvpoint;
			};
			float r;
		};
		float arr[3];
	};
};

class VPipeline {
	friend class VisionTool;
public:
	inline const std::string& getName() const { return this->name; }

protected:
	inline static cv::Mat_<float>
		default_matrix{ cv::Mat_<float>::zeros(3, 3) },
		default_distort{ cv::Mat_<float>::zeros(1, 5) }
	;

	VPipeline() = delete;
	VPipeline(const VPipeline&) = delete;
	inline VPipeline(const std::string& name) : name{name} {}
	inline VPipeline(std::string&& name) : name{std::move(name)} {}

	virtual void invokeGui();
	virtual void process(cv::Mat& io_frame) = 0;
	inline void process(cv::Mat& io_frame, cv::Mat1f* m, cv::Mat1f* c) {
		this->calib_matrix = m;
		this->calib_coeffs = c;
		this->process(io_frame);
	}

private:
	std::string name;
	const cv::Mat_<float>
		*calib_matrix{ &default_matrix },
		*calib_coeffs{ &default_distort };
	bool s_window_enabled{false};

};

class Analyzer : public VPipeline {
public:
	inline Analyzer() : VPipeline{"Test Analyzer"} {}
	~Analyzer() = default;

	virtual void invokeGui() override;
	virtual void process(cv::Mat& io_frame) override;


protected:
	inline void resizeBuffers(Vec2f s) {
		this->proc = cv::Mat{ s.cvsize, CV_8UC3 };
		this->binary = cv::Mat{ s.cvsize, CV_8UC1 };
		this->channels = {
			cv::Mat(s.cvsize, CV_8UC1),
			cv::Mat(s.cvsize, CV_8UC1),
			cv::Mat(s.cvsize, CV_8UC1)
		};
	}

	cv::Mat
		proc,
		binary;
	std::array<cv::Mat, 3> channels;

	std::vector<int> ids;
	std::vector<cv::Vec3f> circles;
	std::vector<std::vector<cv::Point2f> > corners;
	std::vector<std::vector<cv::Point2i> > contours;
	std::vector<cv::Point2f> plot;
	
	static inline cv::Ptr<cv::aruco::Dictionary>
		markers_dict{
			cv::aruco::generateCustomDictionary(7, 4, cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50))
	};

	float
		alpha{ 0.75f }, beta{ 0.75f }, gamma{ 20.f },
		area_frame{0.006f},
		thresh_concavity{0.8},
		dp{ 2.f },
		min_dist{ 10.f },
		hough_p1{ 30.f },	// canny thresh
		hough_p2{ 30.f }	// accumulator thresh
	;
	int
		hue_rotate{ 165 },
		blur{ 5 },
		min_rad{ 5 },
		max_rad{ 20 }
	;


};
class CalibAruco : public VPipeline {
public:
	inline CalibAruco() :
		VPipeline{"Calibration and Aruco Tool"},
		marker_dict{ cv::aruco::getPredefinedDictionary(0) }
	{ this->recreateBoard(); }
	~CalibAruco() = default;

	virtual void invokeGui() override;
	virtual void process(cv::Mat& io_frame) override;

	void calibRoutine(double* a_re = nullptr, double* ch_re = nullptr);
	inline void recreateBoard() {
		this->ch_board = cv::aruco::CharucoBoard::create(
			this->sq_x, this->sq_y,
			this->sq_len, this->marker_len,
			this->marker_dict
		);
		this->board = this->ch_board.staticCast<cv::aruco::Board>();
	}
	inline void resetCollection() {
		const std::lock_guard<std::mutex> vec_lock(this->buff_mutex);
		this->corners.resize(1);
		this->ids.resize(1);
		this->imgs.clear();
		this->keyframes = 0;
	}
	static inline void calibThread(CalibAruco* v) {
		std::lock_guard<std::mutex> l(v->buff_mutex);
		v->calibRoutine();
		v->s_calib_thread = true;
	}

protected:
	cv::Ptr<cv::aruco::Dictionary> marker_dict;
	cv::Ptr<cv::aruco::CharucoBoard> ch_board;
	cv::Ptr<cv::aruco::Board> board;	// or just convert inline when needed

	struct Anotation {
		std::vector<std::vector<cv::Point2f> > corners;
		std::vector<int> ids;
		cv::Mat img, rvec, tvec;
	};

	std::vector<std::vector<std::vector<cv::Point2f> > > corners{1};
	std::vector<std::vector<int> > ids{1};
	std::vector<cv::Mat> imgs, rvecs, tvecs, ch_corners{1}, ch_ids{1};

	std::vector<std::vector<cv::Point2f> > corners_concat, rejected;
	std::vector<int> ids_concat, count_per_frame;
	cv::Mat cam_matx, cam_dist;
	size_t keyframes{0};

	std::mutex buff_mutex;
	std::thread calib_thread;

	std::string dict_custom;
	union {
		struct { int sq_x, sq_y; };
		int sq_arr[2]{5, 7};
	};
	union {
		struct { float sq_len, marker_len; };
		float len_arr[2]{ 1.f, 0.5f };
	};
	int
		fixed_ar_wh[2]{ 100, 100 },
		calib_flags{ 0 },
		dict_id{0},
		pixels_per_unit{96},
		marker_thresh{3},
		ch_point_thresh{6}
	;
	float
		page_size_wh[2]{8.5f, 11.f}
	;
	bool
		s_apply_refined{false},
		s_zero_tan_distort{false},
		s_fix_aspect_ratio{false},
		s_fix_principle_center{false},

		s_auto_collection{false},
		s_keep_next{false},
		s_valid_calib{false},
		s_calib_thread{false}
	;

	inline static const char* dict_names[22]{
		"4x4-50", "4x4-100", "4x4-250", "4x4-1000",
		"5x5-50", "5x5-100", "5x5-250", "5x5-1000",
		"6x6-50", "6x6-100", "6x6-250", "6x6-1000",
		"7x7-50", "7x7-100", "7x7-250", "7x7-1000",
		"Original-1024",
		"AprilTag-16h5", "AprilTag-25h9", "AprilTag-36h10", "AprilTag-36h11",
		"Custom"
	};


};

class VisionTool : public Walnut::Layer {
public:
	inline VisionTool() : proc_thread{imgPipeline, this, std::cref(this->s_thread)} {}
	inline ~VisionTool() {
		this->s_thread = false;
		this->proc_thread.join();
	}

	void addPipeline(VPipeline*);
	void addPipelines(std::vector<VPipeline*>&&);
	void addPipelines(std::initializer_list<VPipeline*>);

	virtual void OnUIRender() override;
	inline void invokeMenuPresence() {
		if (ImGui::BeginMenu("Tools")) {
			ImGui::MenuItem("Vision", NULL, &this->s_tool_enable);
			ImGui::EndMenu();
		}
	}

protected:
	static void imgPipeline(VisionTool*, const bool&);
	inline void resizeViewPortBuffers(Vec2f s) {
		this->viewport_matbuff = cv::Mat{ s.cvsize, CV_8UC4 };
		this->viewport_gmem->Resize(s.width, s.height);
	}

private:
	std::vector<VPipeline*> pipelines;
	size_t idx{ 0 };

	cv::VideoCapture cap_src;
	cv::Mat
		raw_frame,			// 3C BGR	Size of input
		rgba_out,			// 4C RGBA	^
		viewport_matbuff	// 4C RGBA	Size calulcated for output
	;
	const std::shared_ptr<Walnut::Image>
		viewport_gmem{ std::make_shared<Walnut::Image>(0, 0, Walnut::ImageFormat::RGBA) };

	Vec2f view_size{0}, frame_size{0}, raw_size{0};
	float fps;
	int fps_override{-1};
	//std::mutex framebuff_mutex;

	std::thread proc_thread;

	std::string vfile;
	// states
		bool s_tool_enable{ true };

		bool s_capturing{ false };
		bool s_playing{ false };
		
		bool s_thread{ true };
		
		bool s_loopmode{ true };
		bool s_upscale_viewport{ false };
		bool s_grabnext{ false };
		bool s_limit_frate{ false };

		bool s_enable_proc{ false };
		bool s_fps_display{ true };


};


//class AppLayer : public Walnut::Layer {
//public:
//	AppLayer() {}
//	~AppLayer() {}
//
//	//virtual void OnAttach() override;
//	//virtual void OnDetach() override;
//	virtual void OnUIRender() override;
//	void menuFunctionality();
//
//
//private:
//	void resizeImageBuffers(cv::Size);
//
//	cv::VideoCapture vsrc;
//	cv::Mat cap, proc, binary, rgba, marker, aruco_buff{cv::Size(180, 180), CV_8UC4};
//	std::array<cv::Mat, 3> channels;
//	cv::Size framesize;
//	std::vector<int> ids;
//	std::vector<std::vector<cv::Point2f> > corners, contours;
//	//cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
//	static inline cv::Ptr<cv::aruco::Dictionary>
//		markers_dict{
//			cv::aruco::generateCustomDictionary(7, 4, cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50))
//		};
//	static inline struct PrintBoard : public cv::Ptr<cv::aruco::GridBoard> {
//		inline PrintBoard() : cv::Ptr<cv::aruco::GridBoard>{ cv::aruco::GridBoard::create(5, 7, 3, 1, markers_dict) } {
//			(*this)->setIds(std::vector<int>{
//				0, 0, 0, 0, 0,
//				1, 1, 1, 1, 1, 
//				2, 2, 2, 2, 2, 
//				3, 3, 3, 3, 3, 
//				4, 4, 4, 4, 4, 
//				5, 5, 5, 5, 5, 
//				6, 6, 6, 6, 6
//			});
//		}
//	} markers_export;
//
//	std::shared_ptr<Walnut::Image>
//		stream_framebuff{std::make_shared<Walnut::Image>(0, 0, Walnut::ImageFormat::RGBA)},
//		aruco_view{std::make_shared<Walnut::Image>(180, 180, Walnut::ImageFormat::RGBA)}
//	;
//	ImVec2 viewsize;
//	//NeuralNetwork nnet;
//	bool
//		vision_w{true},
//		data_w{false},
//		nn_w{false},
//		demo_w{false},
//
//		capturing{false},
//		stream_enable{true}
//	;
//	int id{0};
//	float alpha{ 0.4f }, beta{ 0.6f }, gamma{ 50.f };
//	std::string vfile, sfile;
//
//	//static void train_thread(AppLayer*);
//
//};