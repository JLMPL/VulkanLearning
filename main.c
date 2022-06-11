#define _CRT_SECURE_NO_WARNINGS

#include "vulkan_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define INVALID_INDEX -1

#define ZERO_OUT(x) memset(&(x), 0, sizeof(x))

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

#define MAX_FRAMES_IN_FLIGHT 2

#define NUM_VALIDATION_LAYERS 1
const char* validationLayers[NUM_VALIDATION_LAYERS] =
{
	"VK_LAYER_KHRONOS_validation"
};

#define NUM_DEVICE_EXTENSIONS 1
const char* deviceExtensions[NUM_DEVICE_EXTENSIONS] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//#ifdef NDEBUG
//const bool enableValidationLayers = false;
//#else
const bool enableValidationLayers = true;
//#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
	printf("validation layer: %s\n", pCallbackData->pMessage);
	return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if (func != NULL)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (func != NULL)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

typedef struct QueueFamilyIndices
{
	int32_t graphicsFamily;
	int32_t presentFamily;
}QueueFamilyIndices;

static bool QueueFamilyIndices_isComplete(const QueueFamilyIndices* qi)
{
	return (qi->graphicsFamily != INVALID_INDEX) && (qi->presentFamily != INVALID_INDEX);
}

typedef struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;

	VkSurfaceFormatKHR formats[64];
	int num_formats;
	VkPresentModeKHR presentModes[64];
	int num_presentModes;
}SwapChainSupportDetails;

static GLFWwindow* window = VK_NULL_HANDLE;

static VkInstance instance = VK_NULL_HANDLE;
static VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
static VkSurfaceKHR surface = VK_NULL_HANDLE;

static VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
static VkDevice device;

static VkQueue graphicsQueue;
static VkQueue presentQueue;

static VkSwapchainKHR swapChain;
static VkFormat swapChainImageFormat;
static VkExtent2D swapChainExtent;

static int num_swapchain_images = 0;
static VkImage swapChainImages[8];
static VkImageView swapChainImageViews[8];
static VkFramebuffer swapChainFramebuffers[8];

static VkCommandPool commandPool;
VkCommandBuffer commandBuffers[8];

static VkFence imagesInFlight[8];

static VkRenderPass renderPass;
static VkDescriptorSetLayout descriptorSetLayout;
static VkPipelineLayout pipelineLayout;
static VkPipeline graphicsPipeline;

static VertexBuffer vertexBuffer;
static VertexBuffer indexBuffer;

static VkDescriptorPool descriptorPool;
static VertexBuffer uniformBuffers[MAX_FRAMES_IN_FLIGHT];
static VkDescriptorSet descriptorSets[MAX_FRAMES_IN_FLIGHT];

static Image textureImage;
static VkImageView textureImageView;
static VkSampler textureSampler;

static VkSemaphore imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT];
static VkSemaphore renderFinishedSemaphores[MAX_FRAMES_IN_FLIGHT];
static VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];
static size_t currentFrame = 0;

static bool framebufferResized = false;

#define NUM_VERTS 4
static vertex_t verts[NUM_VERTS] =
{
	{{-1,-1,0}, {0,0,1}, {0,0}},
	{{1,-1,0}, {1,1,1}, {1,0}},
	{{-1,1,0}, {1,0,0}, {0,1}},
	{{1,1,0}, {1,1,0}, {1,1}}
};

#define NUM_INDS 6
static uint16_t indices[NUM_INDS] =
{
	0,1,2, 2,1,3
};

static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	framebufferResized = true;
}

static void initWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", NULL, NULL);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

static void cleanupSwapChain()
{
	for (int i = 0; i < num_swapchain_images; i++)
	{
		vkDestroyFramebuffer(device, swapChainFramebuffers[i], NULL);
	}

	// vkFreeCommandBuffers(device, commandPool, num_swapchain_images, commandBuffers);

	vkDestroyPipeline(device, graphicsPipeline, NULL);
	vkDestroyPipelineLayout(device, pipelineLayout, NULL);
	vkDestroyRenderPass(device, renderPass, NULL);

	for (int i = 0; i < num_swapchain_images; i++)
	{
		vkDestroyImageView(device, swapChainImageViews[i], NULL);
	}

	vkDestroySwapchainKHR(device, swapChain, NULL);
}

static void cleanup()
{
	cleanupSwapChain();

	vkDestroySampler(device, textureSampler, NULL);
	vkDestroyImageView(device, textureImageView, NULL);

	vkDestroyImage(device, textureImage.image, NULL);
	vkFreeMemory(device, textureImage.memory, NULL);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyBuffer(device, uniformBuffers[i].buffer, NULL);
		vkFreeMemory(device, uniformBuffers[i].memory, NULL);
	}

	vkDestroyDescriptorPool(device, descriptorPool, NULL);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);

	vkDestroyBuffer(device, vertexBuffer.buffer, NULL);
	vkFreeMemory(device, vertexBuffer.memory, NULL);

	vkDestroyBuffer(device, indexBuffer.buffer, NULL);
	vkFreeMemory(device, indexBuffer.memory, NULL);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(device, renderFinishedSemaphores[i], NULL);
		vkDestroySemaphore(device, imageAvailableSemaphores[i], NULL);
		vkDestroyFence(device, inFlightFences[i], NULL);
	}

	vkDestroyCommandPool(device, commandPool, NULL);

	vkDestroyDevice(device, NULL);

	if (enableValidationLayers)
	{
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
	}

	vkDestroySurfaceKHR(instance, surface, NULL);
	vkDestroyInstance(instance, NULL);

	glfwDestroyWindow(window);

	glfwTerminate();
}

static bool checkValidationLayerSupport()
{
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, NULL);

	VkLayerProperties availableLayers[512];
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

	for (int i = 0; i < NUM_VALIDATION_LAYERS; i++)
	{
		bool layerFound = false;

		for (int j = 0; j < layerCount; j++)
		{
			if (strcmp(validationLayers[i], availableLayers[j].layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
		{
			return false;
		}
	}

	return true;
}

const char** getRequiredExtensions(int* count)
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	*count = glfwExtensionCount;
	return glfwExtensions;
}

static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT* createInfo)
{
	ZERO_OUT(*createInfo);
	createInfo->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo->messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo->pfnUserCallback = debugCallback;
}

static void createInstance()
{
	if (enableValidationLayers && !checkValidationLayerSupport())
	{
		printf("validation layers requested, but not available!\n");
	}

	VkApplicationInfo appInfo;
	ZERO_OUT(appInfo);
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Hello Triangle";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "Big Dick Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo;
	ZERO_OUT(createInfo);
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	int extensionCount = 0;
	const char** extensions = getRequiredExtensions(&extensionCount);
	createInfo.enabledExtensionCount = extensionCount;
	createInfo.ppEnabledExtensionNames = extensions;

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
	ZERO_OUT(debugCreateInfo);

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = NUM_VALIDATION_LAYERS;
		createInfo.ppEnabledLayerNames = validationLayers;

		populateDebugMessengerCreateInfo(&debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else
	{
		createInfo.enabledLayerCount = 0;

		createInfo.pNext = NULL;
	}

	if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS)
	{
		printf("failed to create instance!\n");
	}
}

static void setupDebugMessenger()
{
	if (!enableValidationLayers) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo;
	populateDebugMessengerCreateInfo(&createInfo);

	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, NULL, &debugMessenger) != VK_SUCCESS)
	{
		printf("failed to set up debug messenger!\n");
	}
}

static void createSurface()
{
	if (glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS)
	{
		printf("failed to create window surface!\n");
	}
}

static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
{
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);

	VkQueueFamilyProperties queueFamilies[32];
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

	for (int i = 0; i < queueFamilyCount; i++)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			indices.graphicsFamily = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (presentSupport)
		{
			indices.presentFamily = i;
		}

		if (QueueFamilyIndices_isComplete(&indices))
		{
			break;
		}
	}

	return indices;
}

#define MAX_AVALIABLE_EXTENSIONS 512

static bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	uint32_t extensionCount;
	VkExtensionProperties availableExtensions[MAX_AVALIABLE_EXTENSIONS];
	vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);

	vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, availableExtensions);

	int matches = 0;
	for (int i = 0; i < NUM_DEVICE_EXTENSIONS; i++)
	{
		for (int j = 0; j < extensionCount; j++)
		{
			if (strcmp(deviceExtensions[i], availableExtensions[j].extensionName) == 0)
			{
				matches++;
			}
		}
	}

	return matches == NUM_DEVICE_EXTENSIONS;
}

static SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
{
	SwapChainSupportDetails details;
	memset(&details, 0, sizeof(SwapChainSupportDetails));

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, NULL);

	if (formatCount != 0)
	{
		details.num_formats = formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats);
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, NULL);

	if (presentModeCount != 0)
	{
		details.num_presentModes = presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes);
	}

	return details;
}

static bool isDeviceSuitable(VkPhysicalDevice device)
{
	QueueFamilyIndices indices = findQueueFamilies(device);

	bool extensionsSupported = checkDeviceExtensionSupport(device);

	bool swapChainAdequate = false;
	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = swapChainSupport.num_formats && swapChainSupport.num_presentModes;
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

	return QueueFamilyIndices_isComplete(&indices) && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

static void pickPhysicalDevice()
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

	if (deviceCount == 0)
	{
		printf("failed to find GPUs with Vulkan support!\n");
	}

	VkPhysicalDevice devices[32];
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

	for (int i = 0; i < deviceCount; i++)
	{
		if (isDeviceSuitable(devices[i]))
		{
			physicalDevice = devices[i];
			break;
		}
	}

	if (physicalDevice == VK_NULL_HANDLE)
	{
		printf("failed to find a suitable GPU!\n");
	}
}

static void createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	VkDeviceQueueCreateInfo queueCreateInfos[2];
	int32_t uniqueQueueFamilies[] = {indices.graphicsFamily, indices.presentFamily};

	float queuePriority = 1.0f;
	for (int i = 0; i < 2; i++)
	{
		VkDeviceQueueCreateInfo queueCreateInfo;
		ZERO_OUT(queueCreateInfo);
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = uniqueQueueFamilies[i];
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos[i] = queueCreateInfo;
	}

	VkPhysicalDeviceFeatures deviceFeatures;
	ZERO_OUT(deviceFeatures);
	deviceFeatures.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo createInfo;
	ZERO_OUT(createInfo);
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	createInfo.queueCreateInfoCount = 2;
	createInfo.pQueueCreateInfos = queueCreateInfos;

	createInfo.pEnabledFeatures = &deviceFeatures;

	createInfo.enabledExtensionCount = NUM_DEVICE_EXTENSIONS;
	createInfo.ppEnabledExtensionNames = deviceExtensions;

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = NUM_VALIDATION_LAYERS;
		createInfo.ppEnabledLayerNames = validationLayers;
	}
	else
	{
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) != VK_SUCCESS)
	{
		printf("failed to create logical device!\n");
	}

	vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
	vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(SwapChainSupportDetails* details)
{
	for (int i = 0; i < details->num_formats; i++)
	{
		if (details->formats[i].format == VK_FORMAT_B8G8R8A8_SRGB && details->formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return details->formats[i];
		}
	}

	return details->formats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(SwapChainSupportDetails* details)
{
	for (int i = 0; i < details->num_presentModes; i++)
	{
		if (details->presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return details->presentModes[i];
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

static int clampi(int value, int min, int max)
{
	if (value < min)
		return min;

	if (value > max)
		return max;

	return value;
}

static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR* capabilities)
{
	if (capabilities->currentExtent.width != UINT32_MAX)
	{
		return capabilities->currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent =
		{
			width,
			height
		};

		actualExtent.width = clampi(actualExtent.width, capabilities->minImageExtent.width, capabilities->maxImageExtent.width);
		actualExtent.height = clampi(actualExtent.height, capabilities->minImageExtent.height, capabilities->maxImageExtent.height);

		return actualExtent;
	}
}

static void createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(&swapChainSupport);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(&swapChainSupport);
	VkExtent2D extent = chooseSwapExtent(&swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo;
	ZERO_OUT(createInfo);
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;

	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamily, indices.presentFamily };

	if (indices.graphicsFamily != indices.presentFamily)
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;

	if (vkCreateSwapchainKHR(device, &createInfo, NULL, &swapChain) != VK_SUCCESS)
	{
		printf("failed to create swap chain!\n");
	}

	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, NULL);
	num_swapchain_images = imageCount;
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages);

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
}

static VkImageView createImageView(VkImage image, VkFormat format)
{
	VkImageViewCreateInfo viewInfo;
	ZERO_OUT(viewInfo);
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	if (vkCreateImageView(device, &viewInfo, NULL, &imageView) != VK_SUCCESS)
	{
		printf("failed to create image view!\n");
	}

	return imageView;
}

static void createImageViews()
{
	for (size_t i = 0; i < num_swapchain_images; i++)
	{
		swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
	}
}

static void createRenderPass()
{
	VkAttachmentDescription colorAttachment;
	ZERO_OUT(colorAttachment);
	colorAttachment.format = swapChainImageFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef;
	ZERO_OUT(colorAttachmentRef);
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass;
	ZERO_OUT(subpass);
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkSubpassDependency dependency;
	ZERO_OUT(dependency);
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPassInfo;
	ZERO_OUT(renderPassInfo);
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	if (vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass) != VK_SUCCESS)
	{
		printf("failed to create render pass!\n");
	}
}

typedef struct
{
	char* buffer;
	size_t size;
}shaderSource_t;

static shaderSource_t readFile(const char* filename)
{
	FILE* file = fopen(filename, "rb");

	if (!file)
	{
		printf("failed to open file %s!\n", filename);
	}

	shaderSource_t source;

	fseek(file, 0, SEEK_END);
	source.size = ftell(file);
	source.buffer = (char*)malloc(source.size);
	fseek(file, 0, SEEK_SET);
	fread(source.buffer, source.size, 1, file);

	fclose(file);
	return source;
}

static VkShaderModule createShaderModule(shaderSource_t code)
{
	VkShaderModuleCreateInfo createInfo;
	ZERO_OUT(createInfo);
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size;
	createInfo.pCode = (const uint32_t*)(code.buffer);

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, NULL, &shaderModule) != VK_SUCCESS)
	{
		printf("failed to create shader module!\n");
	}

	return shaderModule;
}

static VkVertexInputBindingDescription Vertex_getBindingDescription()
{
	VkVertexInputBindingDescription bindingDesc;
	ZERO_OUT(bindingDesc);
	bindingDesc.binding = 0;
	bindingDesc.stride = sizeof(vertex_t);
	bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	return bindingDesc;
}

typedef struct
{
	VkVertexInputAttributeDescription descs[3];
}vertexInputAttributeDescriptions_t;

static vertexInputAttributeDescriptions_t Vertex_getAttributeDescriptions()
{
	vertexInputAttributeDescriptions_t viad;
	viad.descs[0].binding = 0;
	viad.descs[0].location = 0;
	viad.descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	viad.descs[0].offset = offsetof(vertex_t, pos);

	viad.descs[1].binding = 0;
	viad.descs[1].location = 1;
	viad.descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	viad.descs[1].offset = offsetof(vertex_t, color);

	viad.descs[2].binding = 0;
	viad.descs[2].location = 2;
	viad.descs[2].format = VK_FORMAT_R32G32_SFLOAT;
	viad.descs[2].offset = offsetof(vertex_t, coord);

	return viad;
}

static void createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding uboLayoutBinding;
	ZERO_OUT(uboLayoutBinding);
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	uboLayoutBinding.pImmutableSamplers = NULL;

	VkDescriptorSetLayoutBinding samplerLayoutBinding;
	ZERO_OUT(samplerLayoutBinding);
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.pImmutableSamplers = NULL;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutBinding bindings[2] = {uboLayoutBinding, samplerLayoutBinding};
	VkDescriptorSetLayoutCreateInfo layoutInfo;
	ZERO_OUT(layoutInfo);
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 2;
	layoutInfo.pBindings = bindings;

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout) != VK_SUCCESS)
	{
		printf("failed to create descriptor set!\n");
	}
}

static void createGraphicsPipeline()
{
	shaderSource_t vertShaderCode = readFile("data/vert.spv");
	shaderSource_t fragShaderCode = readFile("data/frag.spv");

	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	free(vertShaderCode.buffer);
	free(fragShaderCode.buffer);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo;
	ZERO_OUT(vertShaderStageInfo);
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo;
	ZERO_OUT(fragShaderStageInfo);
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	VkVertexInputBindingDescription bindingDesc = Vertex_getBindingDescription();
	vertexInputAttributeDescriptions_t attribDesc = Vertex_getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo;
	ZERO_OUT(vertexInputInfo);
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = 2;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
	vertexInputInfo.pVertexAttributeDescriptions = attribDesc.descs;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly;
	ZERO_OUT(inputAssembly);
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport;
	ZERO_OUT(viewport);
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)swapChainExtent.width;
	viewport.height = (float)swapChainExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	ZERO_OUT(scissor);
	scissor.offset = (VkOffset2D){ 0, 0 };
	scissor.extent = swapChainExtent;

	VkPipelineViewportStateCreateInfo viewportState;
	ZERO_OUT(viewportState);
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer;
	ZERO_OUT(rasterizer);
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisampling;
	ZERO_OUT(multisampling);
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState colorBlendAttachment;
	ZERO_OUT(colorBlendAttachment);
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo colorBlending;
	ZERO_OUT(colorBlending);
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkPipelineLayoutCreateInfo pipelineLayoutInfo;
	ZERO_OUT(pipelineLayoutInfo);
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout) != VK_SUCCESS)
	{
		printf("failed to create pipeline layout!\n");
	}

	VkGraphicsPipelineCreateInfo pipelineInfo;
	ZERO_OUT(pipelineInfo);
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &graphicsPipeline) != VK_SUCCESS)
	{
		printf("failed to create graphics pipeline!\n");
	}

	vkDestroyShaderModule(device, fragShaderModule, NULL);
	vkDestroyShaderModule(device, vertShaderModule, NULL);
}

static void createFramebuffers()
{
	for (size_t i = 0; i < num_swapchain_images; i++)
	{
		VkImageView attachments[] =
		{
			swapChainImageViews[i]
		};

		VkFramebufferCreateInfo framebufferInfo;
		ZERO_OUT(framebufferInfo);
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapChainExtent.width;
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, NULL, &swapChainFramebuffers[i]) != VK_SUCCESS)
		{
			printf("failed to create framebuffer!\n");
		}
	}
}

static void createCommandPool()
{
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

	VkCommandPoolCreateInfo poolInfo;
	ZERO_OUT(poolInfo);
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

	if (vkCreateCommandPool(device, &poolInfo, NULL, &commandPool) != VK_SUCCESS)
	{
		printf("failed to create command pool!\n");
	}
}

static int findMemoryType(int typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (int i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	printf("Could not find suitable memory type for VkBuffer!\n");
}

static VertexBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
	VertexBuffer vb;
	ZERO_OUT(vb);

	VkBufferCreateInfo bufferInfo;
	ZERO_OUT(bufferInfo);
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(device, &bufferInfo, NULL, &vb.buffer) != VK_SUCCESS)
	{
		printf("Could not create VkBuffer!\n");
	}

	VkMemoryRequirements memRequriements;
	vkGetBufferMemoryRequirements(device, vb.buffer, &memRequriements);

	VkMemoryAllocateInfo allocInfo;
	ZERO_OUT(allocInfo);
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequriements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequriements.memoryTypeBits, properties);

	if (vkAllocateMemory(device, &allocInfo, NULL, &vb.memory) != VK_SUCCESS)
	{
		printf("Could not allocate VkBuffer memory!\n");
	}

	vkBindBufferMemory(device, vb.buffer, vb.memory, 0);
	return vb;
}

static VkCommandBuffer beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo allocInfo;
	ZERO_OUT(allocInfo);
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo;
	ZERO_OUT(beginInfo);
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

static void endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo;
	ZERO_OUT(submitInfo);
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();
	VkBufferCopy copyRegion;
	ZERO_OUT(copyRegion);
	copyRegion.size = size;

	vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

	endSingleTimeCommands(commandBuffer);
}

static Image createImage(
	int width,
	int height,
	VkFormat format,
	VkImageTiling tiling,
	VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties)
{
	Image img;

	VkImageCreateInfo imageInfo;
	ZERO_OUT(imageInfo);
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.flags = 0;

	if (vkCreateImage(device, &imageInfo, NULL, &img.image) != VK_SUCCESS)
	{
		printf("failed to create image!\n");
	}

	VkMemoryRequirements memRequriements;
	vkGetImageMemoryRequirements(device, img.image, &memRequriements);

	VkMemoryAllocateInfo allocInfo;
	ZERO_OUT(allocInfo);
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequriements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequriements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	if (vkAllocateMemory(device, &allocInfo, NULL, &img.memory) != VK_SUCCESS)
	{
		printf("texture alloc failed!\n");
	}

	vkBindImageMemory(device, img.image, img.memory, 0);

	return img;
}

static void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkImageMemoryBarrier barrier;
	ZERO_OUT(barrier);
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	barrier.srcAccessMask = 0; //TODO
	barrier.dstAccessMask = 0; //TODO

	VkPipelineStageFlags srcStage;
	VkPipelineStageFlags dstStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		printf("unsupported layout transition!\n");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		srcStage, dstStage,
		0,
		0, NULL,
		0, NULL,
		1, &barrier
	);

	endSingleTimeCommands(commandBuffer);
}

static void copyBufferToImage(VkBuffer buffer, VkImage image, int width, int height)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region;
	ZERO_OUT(region);
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	//region.imageOffset = {0,0,0};
	region.imageExtent.width = width;
	region.imageExtent.height = height;
	region.imageExtent.depth = 1;

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(commandBuffer);
}

static void createTextureImage()
{
	int width, height, numChannels;

	stbi_uc* pixels = stbi_load("data/test.png", &width, &height, &numChannels, STBI_rgb_alpha);

	VkDeviceSize imageSize = width * height * 4;

	if (!pixels)
	{
		printf("stbi oopsie!\n");
	}

	VertexBuffer stagingBuf = createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	void* data = NULL;
	vkMapMemory(device, stagingBuf.memory, 0, imageSize, 0, &data);
	memcpy(data, pixels, imageSize);
	vkUnmapMemory(device, stagingBuf.memory);

	stbi_image_free(pixels);

	textureImage = createImage(width, height, VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	transitionImageLayout(textureImage.image, VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyBufferToImage(stagingBuf.buffer, textureImage.image, width, height);
	transitionImageLayout(textureImage.image, VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(device, stagingBuf.buffer, NULL);
	vkFreeMemory(device, stagingBuf.memory, NULL);
}

static void createTextureImageView()
{
	textureImageView = createImageView(textureImage.image, VK_FORMAT_B8G8R8A8_SRGB);
}

static void createTextureSampler()
{
	VkSamplerCreateInfo samplerInfo;
	ZERO_OUT(samplerInfo);
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;

	VkPhysicalDeviceProperties properties;
	ZERO_OUT(properties);
	vkGetPhysicalDeviceProperties(physicalDevice, &properties);

	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0;
	samplerInfo.minLod = 0;
	samplerInfo.maxLod = 0;

	if (vkCreateSampler(device, &samplerInfo, NULL, &textureSampler) != VK_SUCCESS)
	{
		printf("could not create texture sampler!\n");
	}
}

static void createVertexBuffer()
{
	VkDeviceSize bufferSize = sizeof(vertex_t) * NUM_VERTS;

	VertexBuffer stagingBuf = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	void* data;
	vkMapMemory(device, stagingBuf.memory, 0, bufferSize, 0, &data);
	memcpy(data, verts, bufferSize);
	vkUnmapMemory(device, stagingBuf.memory);

	vertexBuffer = createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	);

	copyBuffer(stagingBuf.buffer, vertexBuffer.buffer, bufferSize);

	vkDestroyBuffer(device, stagingBuf.buffer, NULL);
	vkFreeMemory(device, stagingBuf.memory, NULL);
}

static void createIndexBuffer()
{
	VkDevice bufferSize = sizeof(uint16_t) * NUM_INDS;

	VertexBuffer stagingBuf = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	void* data;
	vkMapMemory(device, stagingBuf.memory, 0, bufferSize, 0, &data);
	memcpy(data, indices, bufferSize);
	vkUnmapMemory(device, stagingBuf.memory);

	indexBuffer = createBuffer(bufferSize,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	);

	copyBuffer(stagingBuf.buffer, indexBuffer.buffer, bufferSize);

	vkDestroyBuffer(device, stagingBuf.buffer, NULL);
	vkFreeMemory(device, stagingBuf.memory, NULL);
}

static void createUniformBuffers()
{
	VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		uniformBuffers[i] = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	}
}

static void createDescriptorPool()
{
	VkDescriptorPoolSize poolSizes[2];
	memset(poolSizes, 0, sizeof(VkDescriptorPoolSize) * 2);

	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;

	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = MAX_FRAMES_IN_FLIGHT;

	VkDescriptorPoolCreateInfo poolInfo;
	ZERO_OUT(poolInfo);
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

	if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool) != VK_SUCCESS)
	{
		printf("falied to create descriptor pool!\n");
	}
}

static void createDescriptorSets()
{
	VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
	layouts[0] = descriptorSetLayout;
	layouts[1] = descriptorSetLayout;

	VkDescriptorSetAllocateInfo allocInfo;
	ZERO_OUT(allocInfo);
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
	allocInfo.pSetLayouts = layouts;

	if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets))
	{
		printf("failed to allocate descriptor sets!\n");
	}

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo bufferInfo;
		ZERO_OUT(bufferInfo);
		bufferInfo.buffer = uniformBuffers[i].buffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		VkDescriptorImageInfo imageInfo;
		ZERO_OUT(imageInfo);
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = textureImageView;
		imageInfo.sampler = textureSampler;

		VkWriteDescriptorSet descriptorWrites[2];
		memset(descriptorWrites, 0, sizeof(VkWriteDescriptorSet) * 2);

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSets[i];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;
		descriptorWrites[0].pImageInfo = NULL;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &bufferInfo;
		descriptorWrites[1].pImageInfo = &imageInfo;

		vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, NULL);
	}
}

static void createCommandBuffers()
{
	VkCommandBufferAllocateInfo allocInfo;
	ZERO_OUT(allocInfo);
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)num_swapchain_images;

	if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers) != VK_SUCCESS)
	{
		printf("failed to allocate command buffers!\n");
	}
}

static void recordCommandBuffer(VkCommandBuffer commandBuffer, int imageIndex)
{
	VkCommandBufferBeginInfo beginInfo;
	ZERO_OUT(beginInfo);
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
	{
		printf("failed to begin recording command buffer!\n");
	}

	VkRenderPassBeginInfo renderPassInfo;
	ZERO_OUT(renderPassInfo);
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
	renderPassInfo.renderArea.offset = (VkOffset2D){ 0, 0 };
	renderPassInfo.renderArea.extent = swapChainExtent;

	VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
	VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, NULL);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
		
	vkCmdDrawIndexed(commandBuffer, NUM_INDS, 1, 0, 0, 0);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		printf("failed to record command buffer!\n");
	}
}

static void createSyncObjects()
{
	VkSemaphoreCreateInfo semaphoreInfo;
	ZERO_OUT(semaphoreInfo);
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo;
	ZERO_OUT(fenceInfo);
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		if (vkCreateSemaphore(device, &semaphoreInfo, NULL, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, NULL, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fenceInfo, NULL, &inFlightFences[i]) != VK_SUCCESS)
		{
			printf("failed to create synchronization objects for a frame!\n");
		}
	}
}

static void recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	if (vkDeviceWaitIdle(device) != VK_SUCCESS)
	{
		printf("vkDeviceWaitIdle gone bad!\n");
	}

	cleanupSwapChain();

	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createFramebuffers();
}

static float angle = 0.f;

static void updateUniformBuffer(int currentFrame)
{
	UniformBufferObject ubo;
	ubo.proj = MatrixPerspective(1.f, (float)WIDTH/(float)HEIGHT, 0.1f, 25.f);
	ubo.proj.v[1].y *= -1.f;
	ubo.view = MatrixLookAt(Vec3(0,0,5), Vec3(0,0,0), Vec3(0,1,0));
	angle += 0.001f;
	ubo.model = MatrixRotateZ(angle);

	void* data;
	vkMapMemory(device, uniformBuffers[currentFrame].memory, 0, sizeof(UniformBufferObject), 0, &data);
	memcpy(data, &ubo, sizeof(UniformBufferObject));
	vkUnmapMemory(device, uniformBuffers[currentFrame].memory);
}

static void drawFrame()
{
	vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		printf("failed to acquire swap chain image!\n");
	}

	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	updateUniformBuffer(currentFrame);

	vkResetCommandBuffer(commandBuffers[currentFrame], 0);
	recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

	VkSubmitInfo submitInfo;
	ZERO_OUT(submitInfo);
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
	if (result != VK_SUCCESS)
	{
		printf("failed to submit draw command buffer %d!\n", result);
	}

	VkPresentInfoKHR presentInfo;
	ZERO_OUT(presentInfo);
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = NULL;

	result = vkQueuePresentKHR(presentQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
	{
		framebufferResized = false;
		recreateSwapChain();
	}
	else if (result != VK_SUCCESS)
	{
		printf("failed to present swap chain image!\n");
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void initVulkan()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createFramebuffers();
	createCommandPool();
	createTextureImage();
	createTextureImageView();
	createTextureSampler();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
	createSyncObjects();
}

void mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
	}

	if (vkDeviceWaitIdle(device) != VK_SUCCESS)
	{
		printf("vkDeviceWaitIdle gone bad!\n");
	}
}

int main()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}
