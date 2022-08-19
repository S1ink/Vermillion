#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "vision.h"


Walnut::Application* Walnut::CreateApplication(int argc, char** argv) {

	const Walnut::ApplicationSpecification window{ "Vermillion [ALPHA]", 1280, 720 };

	Walnut::Application* app = new Walnut::Application(window);
	std::shared_ptr<VisionTool> vis{ std::make_shared<VisionTool>() };
	app->PushLayer(vis);
	static bool demo{ false };
	app->SetMenubarCallback([app, vis](){
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) {
				app->Close();
			}
			ImGui::MenuItem("Demo", NULL, &demo);
			ImGui::EndMenu();
		}
		vis->invokeMenuPresence();
		if (demo) {
			ImGui::ShowDemoWindow();
		}
	});
	return app;

}