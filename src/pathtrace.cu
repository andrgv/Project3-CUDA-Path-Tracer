#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

// can toggle each feature for performance comparison
#define CONTIGUOUS_BY_MATERIAL 1
#define DIRECT_LIGHTING 1
#define MIS 1
#define RUSSIAN_ROULETTE 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

// predicate for thrust partition
struct hasBouncesRemaining {
    __host__ __device__ bool operator()(const PathSegment &path) const {
        return path.remainingBounces > 0;
    }
};

struct compareMaterial {
    __host__ __device__ bool operator()(
        const ShadeableIntersection &a,
        const ShadeableIntersection &b) const {
            return a.materialId < b.materialId;
        }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// helpers for direct lighting
__host__ __device__
float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    float f2 = f * f;
    float g2 = g * g;

    return f2 + g2 > 0.0f ? f2 / (f2 + g2) : 0.0f;
}

__device__
float TriangleArea(const Geom& triangle) {
    glm::vec3 a = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 b = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    return glm::length(glm::cross(a, b)) / 2.0f;
}

__device__
glm::vec3 TriangleNormal(const Geom& triangle)
{
    glm::vec3 a = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    glm::vec3 b = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    return glm::normalize(glm::cross(a, b));
}

__device__
bool IntersectP(const Ray& ray, float tMax, Geom* geoms, int geoms_size) {
    for (int i = 0; i < geoms_size; ++i) {
        glm::vec3 intersectionPoint;
        glm::vec3 normal;
        bool outside = true;
        float t = -1;

        if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        } else if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        } else if (geoms[i].type == TRIANGLE) {
            t = triangleIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
        }

        if (t > 0 && t < tMax - 0.001) {
            return true;
        }
    }

    return false;
}

__device__
float LightPDF(const Geom& light, int numLights, const Ray& ray, float t) {
    if (numLights == 0) {
        return 0;
    }

    float area = TriangleArea(light);
    float cosLight = glm::max(glm::dot(TriangleNormal(light), -ray.direction), 0.0f);

    if (area <= EPSILON || cosLight <= 0.0f) {
        return 0;
    }

    float lightPMF = 1.0f / numLights;
    return lightPMF * t * t / (area * cosLight);
}

__device__
glm::vec3 SampleLd(
    const glm::vec3& p, const glm::vec3& n, const Material& material,
    Geom* geoms, int geoms_size, Material* materials, int* lightGeomIndices,
    int numLights, thrust::default_random_engine& rng) {
    if (numLights == 0) {
        return glm::vec3(0);
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    int lightIndex = (int)(u01(rng) * numLights);
    if (lightIndex == numLights) {
        lightIndex--;
    }

    const Geom& light = geoms[lightGeomIndices[lightIndex]];
    float area = TriangleArea(light);

    if (area <= EPSILON) {
        return glm::vec3(0.0f);
    }

    float sqrtU = sqrtf(u01(rng));
    float b0 = 1.0f - sqrtU;
    float b1 = u01(rng) * sqrtU;
    float b2 = 1.0f - b0 - b1;

    glm::vec3 pLight =
        b0 * light.triangleVertices[0] +
        b1 * light.triangleVertices[1] +
        b2 * light.triangleVertices[2];

    Ray shadowRay;
    shadowRay.origin = p + 0.001f * n;

    glm::vec3 toLight = pLight - shadowRay.origin;
    float distanceSquared = glm::dot(toLight, toLight);

    if (distanceSquared <= EPSILON) {
        return glm::vec3(0);
    }

    float distance = sqrtf(distanceSquared);
    shadowRay.direction = toLight / distance;

    float cosTheta = glm::max(glm::dot(n, shadowRay.direction), 0.0f);
    float cosLight = glm::max(glm::dot(TriangleNormal(light), -shadowRay.direction), 0.0f);

    if (cosTheta <= 0 || cosLight <= 0 ||
        IntersectP(shadowRay, distance, geoms, geoms_size)) {
        return glm::vec3(0);
    }

    float lightPMF = 1.0f / numLights;
    float p_l = lightPMF * distanceSquared / (area * cosLight);
    float p_b = cosTheta / PI;
    float w_l = 1;

#if MIS
    w_l = PowerHeuristic(1, p_l, 1, p_b);
#endif

    glm::vec3 f = material.color / PI;
    const Material& lightMaterial = materials[light.materialid];
    glm::vec3 Li = lightMaterial.color * lightMaterial.emittance;

    return w_l * Li * f * cosTheta / p_l;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static int *dev_light_geom_indices = NULL;
static int num_lights = 0;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    // initializing all sources of light for direct lighting and MIS

    std::vector<int> lightGeomIndices;
    for (int i = 0; i < scene->geoms.size(); ++i) {
        const Geom &curr_geom = scene->geoms[i];

        if (curr_geom.type != TRIANGLE || curr_geom.materialid < 0 || curr_geom.materialid >= scene->materials.size()) {
            continue;
        }

        if (scene->materials[curr_geom.materialid].emittance > 0) {
            lightGeomIndices.push_back(i);
        }
    }

    num_lights = lightGeomIndices.size();
    if (num_lights > 0) {
        cudaMalloc(&dev_light_geom_indices, num_lights * sizeof(int));
        cudaMemcpy(dev_light_geom_indices, lightGeomIndices.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_light_geom_indices);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1, 1, 1);
        segment.L = glm::vec3(0);
        segment.specularBounce = false;
        segment.p_b = 0;

        // implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitter_x = u01(rng) - 0.5;
        float jitter_y = u01(rng) - 0.5;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    int* lightGeomIndices,
    int numLights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment &path = pathSegments[idx];
        if (path.remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            glm::vec3& L = path.L;
            glm::vec3& beta = path.color;

          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            glm::vec3 p = path.ray.origin + intersection.t * path.ray.direction;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                const Geom& light = geoms[intersection.geomId];
                bool frontFacing = light.type != TRIANGLE ||
                    glm::dot(TriangleNormal(light), -path.ray.direction) > 0;

                if (frontFacing) {
                    glm::vec3 Le = materialColor * material.emittance;

                    #if DIRECT_LIGHTING
                    if (depth == 1 || path.specularBounce || light.type != TRIANGLE) {
                        L += beta * Le;
                    }
                    
                    #if MIS
                    else {
                        float p_l = LightPDF(light, numLights, path.ray, intersection.t);
                        float w_b = PowerHeuristic(1, path.p_b, 1, p_l);
                        L += beta * w_b * Le;
                    }
                    #endif
                    
                    #else
                    L += beta * Le;
                    #endif
                }

                path.remainingBounces = 0;
            }
            else {
                #if DIRECT_LIGHTING
                glm::vec3 Ld = SampleLd(
                    p, intersection.surfaceNormal, material, geoms,
                    geoms_size, materials, lightGeomIndices, numLights, rng);
                L += beta * Ld;
                #endif

                scatterRay(path, p, intersection.surfaceNormal, material, rng);
                path.remainingBounces--;

                float cosTheta = glm::max(
                    glm::dot(intersection.surfaceNormal, path.ray.direction),
                    0.0f);
                path.p_b = cosTheta / PI;
                path.specularBounce = false;

                #if RUSSIAN_ROULETTE

                float maxBeta = glm::max(beta.x, glm::max(beta.y, beta.z));
                if (path.remainingBounces > 0 && maxBeta < 1 && depth > 1) {
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    float result = glm::max(0.0f, 1.0f - maxBeta);
                    if (u01(rng) < result) {
                        path.remainingBounces = 0;
                    }
                    else {
                        beta /= 1 - result;
                    }
                }

                #endif
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.L;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        #if CONTIGUOUS_BY_MATERIAL
        thrust::sort_by_key(
            thrust::device, dev_intersections, 
            dev_intersections + num_paths, dev_paths, 
            compareMaterial());
        #endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_light_geom_indices,
            num_lights
        );
        checkCUDAError("shade material");
        
        PathSegment* activeEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, hasBouncesRemaining());
        num_paths = (int)(activeEnd - dev_paths);
        iterationComplete = !num_paths || depth >= traceDepth; // based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
