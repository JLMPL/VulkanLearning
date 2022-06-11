#ifndef VULKAN_API
#define VULKAN_API
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "raymath.h"

typedef struct
{
	vec3 pos;
	vec3 color;
	vec2 coord;
}vertex_t;

typedef struct
{
	VkBuffer buffer;
	VkDeviceMemory memory;
}VertexBuffer;

typedef struct
{
	VkImage image;
	VkDeviceMemory memory;
}Image;

typedef struct
{
	mat4 proj;
	mat4 view;
	mat4 model;
}UniformBufferObject;

extern VkDevice g_device;
extern VkCommandPool g_commandPool;

VertexBuffer GPU_CreateVertexBuffer(vertex_t* verts, int num_verts);
void GPU_DestroyVertexBuffer(VertexBuffer* vb);

VertexBuffer GPU_CreateIndexBuffer(uint16_t* inds, int num_indices);
void GPU_DestroyIndexBuffer(VertexBuffer* vb);

#endif
