#pragma once

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <thread>

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
	};
};

class VisionTool : public Walnut::Layer {
public:
	inline VisionTool() : proc_thread{imgPipeline, this, std::cref(this->s_thread)} {}
	inline ~VisionTool() {
		this->s_thread = false;
		this->proc_thread.join();
	}

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
	inline void resizeProcBuffers(Vec2f s) {
		this->proc = cv::Mat{ s.cvsize, CV_8UC3 };
		this->binary = cv::Mat{ s.cvsize, CV_8UC1 };
		this->rgba_out = cv::Mat{ s.cvsize, CV_8UC4 };
		this->channels = {
			cv::Mat(s.cvsize, CV_8UC1),
			cv::Mat(s.cvsize, CV_8UC1),
			cv::Mat(s.cvsize, CV_8UC1)
		};
	}

private:
	cv::VideoCapture cap_src;
	cv::Mat
		raw_frame,			// 3C BGR	Size of input
		proc,				// 3C BGR	^
		binary,				// 1C GRAY	^
		rgba_out,			// 4C RGBA	^
		viewport_matbuff	// 4C RGBA	Size calulcated for output
	;
	std::array<cv::Mat, 3> channels;
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f> > corners;
	std::vector<std::vector<cv::Point> > contours;
	const std::shared_ptr<Walnut::Image>
		viewport_gmem{ std::make_shared<Walnut::Image>(0, 0, Walnut::ImageFormat::RGBA) };

	Vec2f view_size{0}, frame_size{0}, raw_size{0};
	float alpha{ 0.4f }, beta{ 0.6f }, gamma{ 50.f };
	float fps;
	//std::mutex framebuff_mutex;

	static inline cv::Ptr<cv::aruco::Dictionary>
		markers_dict{
			cv::aruco::generateCustomDictionary(7, 4, cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50))
		};

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