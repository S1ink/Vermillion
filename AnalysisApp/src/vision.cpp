#include "vision.h"

#include <iostream>

#include "util.h"


inline bool operator!=(ImVec2 a, ImVec2 b) { return a.x != b.x || a.y != b.y; }
template<typename ta, typename tb>
inline bool operator!=(cv::Size_<ta> a, cv::Size_<tb> b) { return a.width != b.width || a.height != b.height; }
inline bool operator>>(ImVec2 a, ImVec2 b) { return a.x > b.x || a.y > b.y; }
inline bool operator<<(ImVec2 a, ImVec2 b) { return a.x < b.x || a.y < b.y; }
template<typename ta, typename tb>
inline bool operator>>(cv::Size_<ta> a, cv::Size_<tb> b) { return a.width > b.width || a.height > b.height; }

inline float noNaN(float v) { return isnan(v) ? 0.f : v; }
inline double noNaN(double v) { return isnan(v) ? 0.0 : v; }


void VPipeline::invokeGui() { ImGui::Text("Default Text - Override 'invokeGui()' to customize input GUI!"); }


void VisionTool::addPipeline(VPipeline* v) { this->pipelines.push_back(v); }
void VisionTool::addPipelines(std::vector<VPipeline*>&& v) {
	this->pipelines.reserve(this->pipelines.size() + v.size());
	this->pipelines.insert(this->pipelines.end(), v.begin(), v.end());
}
void VisionTool::addPipelines(std::initializer_list<VPipeline*> v) {
	this->pipelines.reserve(this->pipelines.size() + v.size());
	this->pipelines.insert(this->pipelines.end(), v.begin(), v.end());
}

void Vision1::invokeGui() {
	ImGui::SliderInt("Hue", &this->hue_rotate, 0, 180);
	ImGui::SliderFloat("Alpha", &this->alpha, 0.f, 1.f, "%.3f");
	ImGui::SliderFloat("Beta", &this->beta, 0.f, 1.f, "%.3f");
	ImGui::SliderFloat("Gamma", &this->gamma, 0.f, 100.f, "%.3f");
	ImGui::SliderInt("Blur", &this->blur, 0, 21);
	ImGui::SliderFloat("DP", &this->dp, 1, 5);
	ImGui::SliderFloat("Min dist %", &this->min_dist, 0.f, 100.f);
	ImGui::SliderFloat("Canny Thresh", &this->hough_p1, 0, 255);
	ImGui::SliderFloat("Acc Thresh", &this->hough_p2, 0, 255);
	ImGui::SliderInt("Min Radius", &this->min_rad, 0, this->max_rad);
	ImGui::SliderInt("Max Radius", &this->max_rad, this->min_rad, this->proc.size().width);
	//ImGui::SliderFloat("Area thresh [% frame]", &this->area_frame, 0.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic);
	//ImGui::SliderFloat("Concavity", &this->thresh_concavity, 0.f, 1.f, "%.3f");
	ImGui::Separator();
	if (ImGui::Button("Add point")) {
		this->plot.emplace_back(this->circles[0][0], this->circles[0][1]);
	}
	if (ImGui::Button("Reset points")) {
		this->plot.clear();
	}

}
void Vision1::process(cv::Mat& io_frame) {
	cv::cvtColor(io_frame, this->proc, cv::COLOR_BGR2HSV);
	for (size_t i = 0; i < this->proc.size().area(); i ++) {
		(this->proc.data[i * 3] += this->hue_rotate) %= 180;
	}
	cv::cvtColor(this->proc, this->proc, cv::COLOR_HSV2BGR);
	cv::split(this->proc, this->channels);
	cv::addWeighted(this->channels[0], this->alpha, this->channels[1], this->beta, this->gamma, this->binary);
	cv::subtract(this->channels[2], this->binary, this->binary);
	if (this->blur > 0 && this->blur % 2 == 1) {
		cv::medianBlur(this->binary, this->binary, this->blur);
	}
	this->circles.clear();
	cv::HoughCircles(this->binary, this->circles, cv::HOUGH_GRADIENT, this->dp, this->min_dist / 100 * this->binary.size().width, this->hough_p1, this->hough_p2, this->min_rad, this->max_rad);

	cv::aruco::detectMarkers(io_frame, this->markers_dict, this->corners, this->ids);
	////cv::findContours(this->binary, this->contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::cvtColor(this->binary, io_frame, cv::COLOR_GRAY2BGR);
	for (size_t i = 0; i < this->circles.size(); i++) {
		cv::circle(io_frame, cv::Point{ (int)this->circles[i][0], (int)this->circles[i][1] }, this->circles[i][2], { 255, 255, 0 }, 2);
		cv::putText(io_frame, std::to_string(i), cv::Point{ (int)this->circles[i][0], (int)this->circles[i][1] }, cv::FONT_HERSHEY_DUPLEX, 0.5, { 0, 255, 0 });
	}
	for (int i = 0; i < (int)this->plot.size() - 1; i++) {
		cv::line(io_frame, this->plot[i], this->plot[i + 1], { 255, 255, 0 });
	}

	//Circle c{0};
	//std::vector< cv::Point2i > contour;
	//for (size_t i = 0; i < this->contours.size(); i++) {
	//	if (cv::contourArea(this->contours[i]) > this->area_frame * io_frame.size().area() / 100) {
	//		cv::drawContours(io_frame, this->contours, i, { 0, 255, 0 }, 2);
	//		cv::minEnclosingCircle(this->contours[i], c.cvpoint, c.r);
	//		cv::convexHull(this->contours[i], contour);
	//		if (cv::contourArea(contour) / (CV_PI * pow(c.r, 2)) > this->thresh_concavity) {
	//			cv::circle(io_frame, c.cvpoint, c.r, { 255, 255, 0 }, 2);
	//		}
	//	} else {
	//		cv::drawContours(io_frame, this->contours, i, { 0, 0, 255 }, 2);
	//	}
	//}
	///*float len = sqrt(this->area_frame * io_frame.size().area() / 100);
	//cv::rectangle(io_frame, cv::Rect2f{100, 100, len, len}, {0, 100, 255}, 2);*/

	cv::aruco::drawDetectedMarkers(io_frame, this->corners, this->ids);
	//cv::aruco::estimatePoseSingleMarkers(this->corners, 1.f, )
}

void VisionTool::OnUIRender() {
	if (this->s_tool_enable) {
		ImGui::Begin("Vision Options"); {
			if (ImGui::Button(this->s_capturing ? "Stop Capture" : "Start Capture")) {
				if (s_capturing = !s_capturing) {
					this->s_playing = false;
					this->cap_src.open(0);
					this->cap_src.set(cv::CAP_PROP_FRAME_WIDTH, 640);
					this->cap_src.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
					this->cap_src.set(cv::CAP_PROP_FPS, 60);
					this->view_size = { 0, 0 };
					std::cout <<
						"Cap mode: " << this->cap_src.get(cv::CAP_PROP_MODE) <<
						"\nFrame height: " << this->cap_src.get(cv::CAP_PROP_FRAME_HEIGHT) <<
						"\nFrame width: " << this->cap_src.get(cv::CAP_PROP_FRAME_WIDTH) <<
						"\nFPS: " << this->cap_src.get(cv::CAP_PROP_FPS) <<
						"\nFrame count: " << this->cap_src.get(cv::CAP_PROP_FRAME_COUNT) <<
						"\nFrame number: " << this->cap_src.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
					this->fps_override = this->cap_src.get(cv::CAP_PROP_FPS);
				}/* else {
					this->cap_src.release();
				}*/
			}
				ImGui::SameLine();
				if (ImGui::Button("Grab Next")) {
					this->s_capturing = false;
					this->s_grabnext = true;
				}
			if (ImGui::Button("Load Video")) {
				if (openFile(this->vfile)) {
					this->s_capturing = false;
					this->cap_src.open(this->vfile);
					this->view_size = { 0, 0 };
					this->s_grabnext = true;
					std::cout <<
						"Cap mode: " << this->cap_src.get(cv::CAP_PROP_MODE) <<
						"\nFrame height: " << this->cap_src.get(cv::CAP_PROP_FRAME_HEIGHT) <<
						"\nFrame width: " << this->cap_src.get(cv::CAP_PROP_FRAME_WIDTH) <<
						"\nFPS: " << this->cap_src.get(cv::CAP_PROP_FPS) <<
						"\nFrame count: " << this->cap_src.get(cv::CAP_PROP_FRAME_COUNT) <<
						"\nFrame number: " << this->cap_src.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
				}
			}
			if (this->vfile.empty()) { ImGui::BeginDisabled(); }
			if (ImGui::Button(this->s_playing ? "Pause" : "Play")) {
				if (this->s_playing = !this->s_playing) {
					this->s_capturing = false;
					if (this->cap_src.get(cv::CAP_PROP_MODE) == -1) {
						this->cap_src.open(this->vfile);
						this->view_size = { 0, 0 };
					}
				}
			}
			if (this->vfile.empty()) { ImGui::EndDisabled(); }

			ImGui::BeginChild("settings", { 0, 0 }, true);
			if (ImGui::Checkbox("Enable Upscale", &this->s_upscale_viewport)) { this->view_size = { 0 }; }
			ImGui::Checkbox("Loop Video", &this->s_loopmode);
			ImGui::Checkbox("Limit Framerate", &this->s_limit_frate);
				ImGui::SameLine();
				ImGui::InputInt("##", &this->fps_override);
			ImGui::Checkbox("Display FPS", &this->s_fps_display);
			ImGui::Checkbox("Enable Processing", &this->s_enable_proc);
			ImGui::Separator();

			if (!this->s_enable_proc) { ImGui::BeginDisabled(); }

			VPipeline* p;
			for (size_t i = 0; i < this->pipelines.size(); i++) {
				p = this->pipelines[i];
				ImGui::Checkbox(p->name.c_str(), &p->s_window_enabled);
					ImGui::SameLine();
					if (ImGui::Button("Set Active")) { this->idx = i; }
				if (p->s_window_enabled) {
					ImGui::Begin(("Vision Pipeline: " + p->name).c_str());
					this->pipelines[i]->invokeGui();
					ImGui::End();
				}
			}

			if (!this->s_enable_proc) { ImGui::EndDisabled(); }

			ImGui::EndChild();
		} ImGui::End();
		ImGui::Begin("Viewport"); {
			//std::lock_guard<std::mutex> io_lock{ this->framebuff_mutex };
			if (this->view_size.imvec != ImGui::GetContentRegionAvail() ||
				this->raw_size.cvsize.aspectRatio() != this->frame_size.cvsize.aspectRatio()
			) {
				this->view_size.imvec = ImGui::GetContentRegionAvail();
				this->frame_size.x = (this->raw_size.x > this->view_size.x || this->s_upscale_viewport ? this->view_size.x : this->raw_size.x);
				this->frame_size.y = noNaN(this->frame_size.x / (float)this->raw_size.cvsize.aspectRatio());
				this->resizeViewPortBuffers(this->frame_size);
				if (!this->rgba_out.empty()) {
					cv::resize(this->rgba_out, this->viewport_matbuff, this->frame_size.cvsize);
				}
			} else if(this->cap_src.isOpened()) {
				cv::resize(this->rgba_out, this->viewport_matbuff, this->frame_size.cvsize);
			}
			if (!this->viewport_matbuff.empty()) {
				this->viewport_gmem->SetData(this->viewport_matbuff.data);
				ImGui::Image(this->viewport_gmem->GetDescriptorSet(), this->frame_size.imvec);
			}
		} ImGui::End();

	}
}
void VisionTool::imgPipeline(VisionTool* that, const bool& state) {
	using hrc = std::chrono::high_resolution_clock;
	hrc::time_point ref;
	//bool keep_current = false;
	for (;state;) {
		if (that->cap_src.isOpened()) {
			if (that->s_playing) {
				if (that->s_loopmode && that->cap_src.get(cv::CAP_PROP_POS_FRAMES) == that->cap_src.get(cv::CAP_PROP_FRAME_COUNT)) {
					that->cap_src.set(cv::CAP_PROP_POS_FRAMES, 0);
				}
				if (!that->cap_src.grab()) { goto wait; }
				if (that->s_limit_frate) {
					std::this_thread::sleep_for(std::chrono::microseconds((uint)(1e6 / (that->fps_override > 0 ? that->fps_override : that->cap_src.get(cv::CAP_PROP_FPS)))) - (hrc::now() - ref));
				}
			}
			else if (that->s_capturing && that->cap_src.grab()) {}
			else if (that->s_grabnext && that->cap_src.grab()) { that->s_grabnext = false; }
			//else if (that->s_enable_proc) { keep_current = true; }
			else { goto wait; }

			that->fps = 1e9 / (hrc::now() - ref).count();
			ref = hrc::now();

			//if (!keep_current) {
				that->cap_src.retrieve(that->raw_frame);
			//	keep_current = false;
				if (that->raw_frame.size() != that->raw_size.cvsize) {
					that->raw_size.cvsize = that->raw_frame.size();
					that->rgba_out = cv::Mat{ that->raw_size.cvsize, CV_8UC4 };
				}
			//} else {
			//	std::this_thread::sleep_for(std::chrono::milliseconds(20) - (hrc::now() - ref));
			//}
			
			if(that->s_enable_proc && that->pipelines.size() > 0) {
				that->pipelines[that->idx]->process(that->raw_frame);
			} 
			cv::cvtColor(that->raw_frame, that->rgba_out, cv::COLOR_BGR2RGBA);

			if (that->s_fps_display) {
				cv::putText(that->rgba_out, "FPS: " + std::to_string(that->fps), { 10, 20 }, cv::FONT_HERSHEY_DUPLEX, 0.5, { 0, 255, 0, 255 });
			}
			continue;
		}
		wait:
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		//std::cout << "FTime: " << static_cast<std::chrono::duration<double>>(hrc::now() - ref).count() << std::endl;
	}
}

//void AppLayer::OnUIRender() {
//	//ImGui::ShowDemoWindow();
//	//ImGui::End();
//	if (this->vision_w) {
//
//		ImGui::Begin("Processing");
//		if (ImGui::Button(this->capturing ? "Stop Capture" : "Start Capture")) {
//			if (capturing = !capturing) {
//				this->vsrc.open(0);
//				this->vsrc.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//				this->vsrc.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//				this->vsrc.set(cv::CAP_PROP_FPS, 60);
//				this->stream_enable = true;
//				this->viewsize = { 0, 0 };
//			}
//			else {
//				this->vsrc.release();
//			}
//		}
//		ImGui::SameLine();
//		if (ImGui::Button("Load Video")) { openFile(this->vfile); }
//		ImGui::Text("Selected file: "); ImGui::SameLine(); ImGui::Text(this->vfile.c_str());
//		if (this->vfile.empty()) {
//			ImGui::BeginDisabled();
//		}
//		if (ImGui::Button("Run Video")) {
//			capturing = false;
//			if (!this->vsrc.open(this->vfile)) {
//				std::cout << "Error: failed to load video file\n";
//			}
//			else {
//				this->viewsize = { 0, 0 };
//				stream_enable = true;
//			}
//		}
//		if (ImGui::Button("Stop Video")) {
//			this->stream_enable = false;
//		}
//		if (this->vfile.empty()) {
//			ImGui::EndDisabled();
//		}
//		
//		ImGui::SliderFloat("Alpha", &this->alpha, 0.f, 1.f, "%.3f", 1);
//		ImGui::SliderFloat("Beta", &this->beta, 0.f, 1.f, "%.3f", 1);
//		ImGui::SliderFloat("Gamma", &this->gamma, 0.f, 100.f, "%.3f", 1);
//		if (ImGui::InputInt("ArUco ID", &this->id)) {
//			if (this->id > 6) { this->id = 6; }
//			if (this->id < 0) { this->id = 0; }
//			cv::aruco::drawMarker(this->markers_dict, this->id, 180, this->marker);
//			cv::cvtColor(this->marker, this->aruco_buff, cv::COLOR_GRAY2RGBA);
//			this->aruco_view->SetData(this->aruco_buff.data);
//		}
//		ImGui::Image(this->aruco_view->GetDescriptorSet(), { 180, 180 });
//		if (ImGui::Button("Save ArUco")) {
//			this->sfile = ("aruco_4x4_" + std::to_string(this->id) + ".png");
//			if (saveFile(this->sfile)) {
//				cv::imwrite(this->sfile, this->marker);
//			}
//		}
//		if (ImGui::Button("Save GridBoard")) {
//			cv::Mat board;
//			this->markers_export->draw({ 816, 1056 }, board, 96);
//			std::string f{ "aruco_board_4x4_5x7.png" };
//			if (saveFile(f)) {
//				cv::imwrite(f, board);
//			}
//		}
//		ImGui::End();
//
//		ImGui::Begin("Video");
//
//		if (this->stream_enable) {
//			if (this->vsrc.isOpened() && this->vsrc.read(this->cap)) {
//
//				if (this->viewsize != ImGui::GetContentRegionAvail()) {
//					this->viewsize = ImGui::GetContentRegionAvail();
//
//					this->framesize = (
//						this->cap.size().aspectRatio() > this->viewsize.x / this->viewsize.y ?
//						(cv::Size2d{ this->viewsize.x, this->viewsize.x / this->cap.size().aspectRatio() }) :
//						(this->cap.size().aspectRatio() < this->viewsize.x / this->viewsize.y ?
//							(cv::Size2d{ this->viewsize.y * this->cap.size().aspectRatio(), this->viewsize.y }) :
//							(cv::Size2f{ this->viewsize.x, this->viewsize.y })
//							));
//					this->resizeImageBuffers(this->framesize);
//				}
//
//				cv::resize(this->cap, this->proc, this->framesize);
//				cv::split(this->proc, this->channels);
//
//				cv::addWeighted(this->channels[0], this->alpha, this->channels[1], this->beta, this->gamma, this->binary);
//				cv::subtract(this->channels[2], this->binary, this->binary);
//
//				cv::aruco::detectMarkers(this->proc, this->markers_dict, this->corners, this->ids);
//
//				cv::cvtColor(this->binary, this->proc, cv::COLOR_GRAY2BGR);
//				cv::aruco::drawDetectedMarkers(this->proc, this->corners, this->ids);
//				cv::cvtColor(this->proc, this->rgba, cv::COLOR_BGR2RGBA);
//
//				this->stream_framebuff->SetData(this->rgba.data);
//			}
//			ImGui::Image(
//				this->stream_framebuff->GetDescriptorSet(),
//				{ (float)this->stream_framebuff->GetWidth(), (float)this->stream_framebuff->GetHeight() }
//			);
//			
//		}
//
//		ImGui::End();
//	}
//	if (this->data_w) {
//		ImGui::Begin("Data");
//		ImGui::End();
//	}
//	if (this->nn_w) {
//		ImGui::Begin("Training");
//		ImGui::End();
//	}
//	if (this->demo_w) {
//		ImGui::ShowDemoWindow();
//	}
//	
//
//}
//void AppLayer::menuFunctionality() {
//	if (ImGui::BeginMenu("File")) {
//		if (ImGui::MenuItem("Save")) {
//			std::cout << "Saved!\n";
//		}
//		ImGui::EndMenu();
//	}
//	if (ImGui::BeginMenu("Tools")) {
//		ImGui::MenuItem("Vision", NULL, &this->vision_w);
//		ImGui::MenuItem("Data", NULL, &this->data_w);
//		ImGui::MenuItem("Train", NULL, &this->nn_w);
//		ImGui::MenuItem("Demo", NULL, &this->demo_w);
//		ImGui::EndMenu();
//	}
//}
//
//
//void AppLayer::resizeImageBuffers(cv::Size nsize) {
//	this->proc = cv::Mat(nsize, CV_8UC3);
//	this->binary = cv::Mat(nsize, CV_8UC1);
//	this->rgba = cv::Mat(nsize, CV_8UC4);
//	this->channels = {
//		cv::Mat(nsize, CV_8UC1),
//		cv::Mat(nsize, CV_8UC1),
//		cv::Mat(nsize, CV_8UC1)
//	};
//	this->stream_framebuff->Resize(nsize.width, nsize.height);
//}