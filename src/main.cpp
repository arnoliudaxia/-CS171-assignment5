#include "cloth.h"
#include "scene.h"

int main() {

  /// settings

  // window
  constexpr int window_width = 1920;
  constexpr int window_height = 1080;

  // cloth
  constexpr Float cloth_weight = Float(1);
  constexpr UVec2 mass_dim = { 40, 30 };
  constexpr Float dx_local = Float(0.1);
  constexpr Float stiffness = Float(100);
  constexpr Float damping_ratio = Float(0.0015);
  std::vector<IVec2> fixed_masses { { 0, -1 }, { -1, -1 } };



  /// setup window
  GLFWwindow* window;
  {
    if (!glfwInit()) // initialize glfw library
      return -1;

    // setting glfw window hints and global configurations
    {
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // use core mode
      // glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE); // use debug context
    #ifdef __APPLE__
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
    #endif
    }

    // create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(window_width, window_height, "CS171 HW5: Cloth Simulation", NULL, NULL);
    if (!window) {
      glfwTerminate();
      return -1;
    }

    // make the window's context current
    glfwMakeContextCurrent(window);

    // load Opengl
    if (!gladLoadGL()) {
      glfwTerminate();
      return -1;
    }

    // setup call back functions
    glfwSetFramebufferSizeCallback(window, Input::CallBackResizeFlareBuffer);
  }

  /// main Loop
  {
    // shader
    Shader::Initialize();

    // scene
    Scene scene(45);
    scene.camera.transform.position = { 0, -1.5, -6 };
    scene.camera.transform.rotation = { 0, 0, 1, 0 };
    scene.light_position = { 0, 3, -10 };
    scene.light_color = Vec3(1, 1, 1) * Float(1.125);

    // clothes
    auto cloth = std::make_shared<RectCloth>(cloth_weight,
                                             mass_dim,
                                             dx_local,
                                             stiffness, damping_ratio);
    for (const auto& fixed_mass : fixed_masses) {
      if (!cloth->SetMassFixedOrNot(fixed_mass.x, fixed_mass.y, true))
        abort();
    }

    // mesh primitives
    auto mesh_cube = std::make_shared<Mesh>(MeshPrimitiveType::cube);
    auto mesh_sphere = std::make_shared<Mesh>(MeshPrimitiveType::sphere);

    // objects
    auto object_cloth = scene.AddObject(cloth,
                                        Shader::shader_phong,
                                        Transform(Vec3(0, 0, 0),
                                                  glm::quat_cast(glm::rotate(Mat4(Float(1.0)), glm::radians(Float(60)), Vec3(1, 0, 0))),
                                                  Vec3(1, 1, 1)));
    auto object_cube = scene.AddObject(mesh_cube,
                                       Shader::shader_phong,
                                       Transform(Vec3(-3.5, -1.8, 0.3),
                                                 Quat(1, 0, 0, 0),
                                                 Vec3(1, 1, 1)));
    auto object_sphere = scene.AddObject(mesh_sphere,
                                         Shader::shader_phong,
                                         Transform(Vec3(3.5, -1.8, 0.3),
                                                   Quat(1, 0, 0, 0),
                                                   Vec3(1, 1, 1)));
    object_cloth->color = { zero, Float(0.75), one };
    object_cube->color = { Float(0.75), one, zero };
    object_sphere->color = { one, Float(0.75), zero };

    // loop until the user closes the window
    Input::Start(window);
//    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
      Input::Update();
      Time::Update();
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      /// terminate
      if (Input::GetKey(KeyCode::Escape))
        glfwSetWindowShouldClose(window, true);

      /// fixed update
      for (unsigned i = 0; i < Time::fixed_update_times_this_frame; ++i) {
        if(Input::GetKey(KeyCode::Space)) { //! only when space is pressed
          scene.FixedUpdate();
        }
//          scene.FixedUpdate();
      }

      /// update
      {
        scene.Update();
//        printf("Pos = (%f, %f, %f)\n", scene.camera.transform.position.x, scene.camera.transform.position.y, scene.camera.transform.position.z);
//        printf("Rot = (%f, %f, %f, %f)\n", scene.camera.transform.rotation.w, scene.camera.transform.rotation.x, scene.camera.transform.rotation.y, scene.camera.transform.rotation.z);
//        printf("\n");
      }

      /// render
      {
        scene.RenderUpdate();
      }

      // swap front and back buffers
      glfwSwapBuffers(window);

      // poll for and process events
      glfwPollEvents();
    }
  }

  glfwTerminate();
  return 0;
}
