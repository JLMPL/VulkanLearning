#ifndef VULKAN_API
#define VULKAN_API
#include <SDL.h>
#include <SDL_vulkan.h>	
#include <vulkan/vulkan.h>
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

#endif
