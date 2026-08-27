#pragma once
#include "scene.h"

namespace adaptive3dgs {

/**
 * @brief Handles rendering of the Scene.
 */
class Renderer {
public:
    Renderer();
    ~Renderer();

    void render(const Scene& scene);

private:
    // TODO: Add renderer state and resources
};

} // namespace adaptive3dgs
