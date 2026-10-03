#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "glm/glm.hpp"

#include <tiny_obj_loader.h>
#include <tiny_gltf_v3.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        loadFromJSON(filename);
        return;
    } else if (ext == ".obj") {
        loadFromOBJ(filename);
    } else if (ext == ".gltf" || ext == ".glb") {
        loadFromGLTF(filename);
    } else {
        cout << "Couldn't read from " << filename << endl;
        return;
    }

    appendMeshToRender();
    initMeshCamera();
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Refractive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGLTF(const std::string& gltfName) {
    tg3_model model{};
    tg3_error_stack errors{};
    tg3_parse_options options{};

    tg3_error_stack_init(&errors);
    tg3_parse_options_init(&options);
    
    const tg3_error_code loaded = tg3_parse_file(
        &model, &errors, gltfName.c_str(), gltfName.length(), &options);
    
    if (loaded != TG3_OK) {
        // error handle
        std::string error_msg = "Error loading glTF: \n";
        for (int i = 0; i < errors.count; ++i) {
            if (errors.entries[i].message) {
                error_msg += "Error " + std::to_string(i) + ": " 
                + errors.entries[i].message + '\n';
            }
        }

        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        
        throw std::runtime_error(error_msg);
    }

    mesh = {};

    const int default_material_id = materials.size();
    Material default_material{};
    default_material.color = glm::vec3(1);
    materials.push_back(default_material);

    const int material_offset = materials.size();

    // load materials
    for (int i = 0; i < model.materials_count; ++i) {
        Material curr_material{};

        curr_material.color = glm::vec3(
            model.materials[i].pbr_metallic_roughness.base_color_factor[0],
            model.materials[i].pbr_metallic_roughness.base_color_factor[1],
            model.materials[i].pbr_metallic_roughness.base_color_factor[2]
        );
        
        curr_material.emittance = glm::max(
            model.materials[i].emissive_factor[0], glm::max(
                model.materials[i].emissive_factor[1],
                model.materials[i].emissive_factor[2]
            ));

        materials.push_back(curr_material);
    }

    for (int i = 0; i < model.meshes_count; ++i) {
        const tg3_mesh curr_mesh = model.meshes[i];
        
        for (int j = 0; j < curr_mesh.primitives_count; ++j) {
            const tg3_primitive &primitive = curr_mesh.primitives[j];
            // load triangles
            if (primitive.mode != TG3_MODE_TRIANGLES) {
                continue;
            }

            const int pos_accessor = findAttribute(primitive, "POSITION");
            if (pos_accessor < 0) {
                continue;
            }

            const int vertex_offset = appendFloatAttribute(
                model, pos_accessor, 3, mesh.vertices);
            if (vertex_offset < 0) {
                continue;
            }

            const int normal_offset = appendFloatAttribute(
                model,
                findAttribute(primitive, "NORMAL"),
                3,
                mesh.normals
            );

            const int texcoord_offset = appendFloatAttribute(
                model,
                findAttribute(primitive, "TEXCOORD_0"),
                2,
                mesh.texcoords
            );

            const unsigned long index_count = appendGLTFIndices(
                model, primitive, vertex_offset, normal_offset, texcoord_offset);

            const int material_id = primitive.material < 0 
                ? default_material_id : material_offset + primitive.material;
        
            for (int i = 0; i < index_count / 3; ++i) {
                mesh.num_face_vertices.push_back(3);
                mesh.material_ids.push_back(material_id);
                mesh.smoothing_group_ids.push_back(0);
            }
        }
    }

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
    
    // success!!
    std::cout << "Success! Loaded GLTF" << std::endl;
}

void Scene::loadFromOBJ(const std::string& objName){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn;
    std::string err;

    std::filesystem::path objPath(objName);
    std::string baseDirectory = objPath.parent_path().string() 
        + std::filesystem::path::preferred_separator;
    
    bool loaded = LoadObj(
        &attrib, &shapes, &objMaterials, &warn, &err, 
        objName.c_str(), baseDirectory.c_str());

    if (!warn.empty()) {
        std::cerr << "Warning: " << warn << std::endl;
    }
    if (!loaded) {
        throw std::runtime_error("Error loading OBJ: " + err);
    }

    mesh.vertices.assign(attrib.vertices.begin(), attrib.vertices.end());
    mesh.normals.assign(attrib.normals.begin(), attrib.normals.end());
    mesh.texcoords.assign(attrib.texcoords.begin(), attrib.texcoords.end());

    // set default material for obj
    Material defaultMaterial{};
    defaultMaterial.color = glm::vec3(1.0f);

    const int defaultMaterialId = this->materials.size();

    this->materials.push_back(defaultMaterial);

    // should be 1
    const int objMaterialOffset = this->materials.size();

    // actually set the right material
    for (const tinyobj::material_t &objMaterial : objMaterials) {
        Material material{};

        const glm::vec3 diffuse(
            objMaterial.diffuse[0],
            objMaterial.diffuse[1],
            objMaterial.diffuse[2]
        );

        const glm::vec3 emission(
            objMaterial.emission[0],
            objMaterial.emission[1],
            objMaterial.emission[2]
        );

        material.emittance = glm::max(
            emission.x, glm::max(emission.y, emission.z)
        );

        // for emissixe materials, color is normalized emission color
        // o.w. storing diffuse color
        if (material.emittance > 0) {
            material.color = emission / material.emittance;
        } else {
            material.color = diffuse;
        }
        this->materials.push_back(material);
    }
    

    for (const tinyobj::shape_t &s : shapes) {
        for (const tinyobj::index_t &i : s.mesh.indices) {
            mesh.indices.push_back(
                {i.vertex_index, i.normal_index, i.texcoord_index}
            );
        }

        mesh.num_face_vertices.insert(
            mesh.num_face_vertices.end(), 
            s.mesh.num_face_vertices.begin(),
            s.mesh.num_face_vertices.end()
        );
            
        for (size_t face = 0; face < s.mesh.num_face_vertices.size(); ++face) {
            const int objMaterialId = face < s.mesh.material_ids.size() ? s.mesh.material_ids[face] : -1;

            if (objMaterialId >= 0 && objMaterialId < objMaterials.size()) {
                mesh.material_ids.push_back(objMaterialOffset + objMaterialId);
            } else {
                mesh.material_ids.push_back(defaultMaterialId);
            }
        }

        mesh.smoothing_group_ids.insert(
            mesh.smoothing_group_ids.end(),
            s.mesh.smoothing_group_ids.begin(),
            s.mesh.smoothing_group_ids.end()
        );
    }

    

    // success!!
    std::cout << "Success! Loaded OBJ" << std::endl;
}

void Scene::appendMeshToRender() {
    size_t offset = 0;
    for (std::size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
        if (mesh.num_face_vertices[f] != 3) {
            // not a triangle, should not get here
            offset += mesh.num_face_vertices[f];
            continue;
        }

        Geom triangle{};
        triangle.type = TRIANGLE;
        triangle.materialid = mesh.material_ids[f];
        triangle.hasNormals = true;

        for (int j = 0; j < 3; ++j) {
            const index_t &index = mesh.indices[offset + j];

            triangle.triangleVertices[j] = glm::vec3(
                mesh.vertices[3 * index.vertex_index],
                mesh.vertices[3 * index.vertex_index + 1],
                mesh.vertices[3 * index.vertex_index + 2]
            );

            if (index.normal_index < 0) {
                triangle.hasNormals = false;
            } else {
                triangle.triangleNormals[j] = glm::vec3(
                    mesh.normals[3 * index.normal_index],
                    mesh.normals[3 * index.normal_index + 1],
                    mesh.normals[3 * index.normal_index + 2]
                );
            }
        }

        geoms.push_back(triangle);
        offset += 3;
    }
}

int Scene::findAttribute(const tg3_primitive &primitive, const char *name) {
    for (int i = 0; i < primitive.attributes_count; ++i) {
        if (tg3_str_equals_cstr(primitive.attributes[i].key, name)) {
            return primitive.attributes[i].value;
        }
    }

    return -1;
}
    
const uint8_t* Scene::getAccessorData(
    const tg3_model &model,
    const tg3_accessor &accessor,
    int32_t &stride
) {
    const tg3_buffer_view &view = model.buffer_views[accessor.buffer_view];
    const tg3_buffer &buffer = model.buffers[view.buffer];
    stride = tg3_accessor_byte_stride(&accessor, &view);
    
    return buffer.data.data + view.byte_offset + accessor.byte_offset;
}

int Scene::appendFloatAttribute(
    const tg3_model &model,
    int accessorIdx,
    int componentCount,
    std::vector<float> &dest
) {
    if (accessorIdx < 0) {
        return -1;
    }

    const tg3_accessor &accessor = model.accessors[accessorIdx];

    if (accessor.component_type != TG3_COMPONENT_TYPE_FLOAT) {
        return -1;
    }

    const int offset = dest.size() / componentCount;
    int32_t stride;
    const uint8_t *data = getAccessorData(model, accessor, stride);

    for (int i = 0; i < accessor.count; ++i) {
        const float *val = (float*) (data + i * stride);

        for (int component = 0; component < componentCount; ++component) {
            dest.push_back(val[component]);
        }
    }

    return offset;
}

unsigned long int Scene::appendGLTFIndices(
    const tg3_model &model,
    const tg3_primitive &primitive,
    int vertex_offset,
    int normal_offset,
    int texcoord_offset
) {
    const tg3_accessor &pos = model.accessors[findAttribute(primitive, "POSITION")];
    const tg3_accessor *index_accessor = nullptr;
    const uint8_t *index_data = nullptr;
    int index_stride = 0;
    uint64_t index_count = pos.count;

    if (primitive.indices >= 0) {
        index_accessor = &model.accessors[primitive.indices];
        index_data = getAccessorData(model, *index_accessor, index_stride);
        index_count = index_accessor->count;
    }

    for (int i = 0; i < index_count; ++i) {
        int idx = i;
        if (index_accessor) {
            const uint8_t *value = index_data + i * index_stride;

            switch (index_accessor->component_type) {
                case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
                    idx = *(uint8_t*)(value);
                    break;
                case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
                    idx = *(uint16_t*)(value);
                    break;
                case TG3_COMPONENT_TYPE_UNSIGNED_INT:
                    idx = *(uint32_t*)(value);
                    break;
                default:
                    throw std::runtime_error("Wrong type for glTF index");
            }
        }
            
        mesh.indices.push_back({
            vertex_offset + idx,
            normal_offset < 0 ? -1 : normal_offset + idx,
            texcoord_offset < 0 ? -1 : texcoord_offset + idx
        });
    }

    return index_count;
}

void Scene::initMeshCamera() {
    if (geoms.empty()) {
        throw std::runtime_error("Mesh contains no triangles");
    }

    glm::vec3 boundsMin(FLT_MAX);
    glm::vec3 boundsMax(-FLT_MAX);

    for (const Geom& geom : geoms) {
        if (geom.type != TRIANGLE) {
            continue;
        }
        for (int i = 0; i < 3; ++i) {
            boundsMin = glm::min(boundsMin, geom.triangleVertices[i]);
            boundsMax = glm::max(boundsMax, geom.triangleVertices[i]);
        }
    }

    const glm::vec3 center = 0.5f * (boundsMin + boundsMax);
    const float radius = glm::max(0.5f * glm::length(boundsMax - boundsMin), 0.001f);

    Camera& camera = state.camera;
    camera.resolution = glm::ivec2(800, 800);

    // setting perspective slightly elevated.
    const glm::vec3 viewDirection = glm::normalize(glm::vec3(1, 0.35, 1));

    const float halfFovY = glm::radians(45.0f);
    const float distance = 1.15 * radius / sin(halfFovY);

    camera.lookAt = center;
    camera.position = center + viewDirection * distance;
    camera.up = glm::vec3(0, 1, 0);
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    const float yScale = tan(halfFovY);
    const float aspect = (float)(camera.resolution.x) / camera.resolution.y;
    const float xScale = yScale * aspect;

    camera.fov = glm::vec2(glm::degrees(atan(xScale)), glm::degrees(halfFovY));

    camera.pixelLength = glm::vec2(
        2 * xScale / camera.resolution.x, 2 * yScale / camera.resolution.y);

    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "mesh";
    state.image.assign(camera.resolution.x * camera.resolution.y, glm::vec3());
}