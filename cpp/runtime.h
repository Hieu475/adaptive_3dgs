#pragma once
#include "scene.h"
#include "renderer.h"

namespace adaptive3dgs {

/**
 * @brief Main runtime loop manager.
 */
class Runtime {
public:
    Runtime();
    ~Runtime();

    void initialize();
    void step();
    void run();

private:
    Scene scene_;
    Renderer renderer_;
    bool is_running_;
};

} // namespace adaptive3dgs
