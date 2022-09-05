#define INCLUDE_IMPLOT

#include <thread>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "implot/implot.h"

#include "vision.h"
#include "mlmodel.h"


void tmain(int argc, char** argv) {
	std::cout << ((-0) == 0) << std::endl;
}


Walnut::Application* Walnut::CreateApplication(int argc, char** argv) {

	//std::thread(tmain, argc, argv).detach();

	const Walnut::ApplicationSpecification window{ "Vermillion [ALPHA]", 1280, 720 };

	Walnut::Application* app = new Walnut::Application(window);
	std::shared_ptr<VisionTool> vis{ std::make_shared<VisionTool>() };
	std::shared_ptr<MlTool> ml{ std::make_shared<MlTool>() };
	static Analyzer pipe;
	static CalibAruco aruco_tool;
	vis->addPipelines({ &aruco_tool, &pipe });
	app->PushLayer(vis);
	app->PushLayer(ml);
	static bool demo{ false };
	app->SetMenubarCallback([app, vis, ml](){
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) {
				app->Close();
			}
			ImGui::MenuItem("Demo", NULL, &demo);
			ImGui::EndMenu();
		}
		vis->invokeMenuPresence();
		ml->invokeMenuPresence();
		if (demo) {
			ImGui::ShowDemoWindow();
			ImPlot::ShowDemoWindow();
		}
	});
	return app;

}