//#include "vulkan_api.h"
//
//VkDevice g_device;
//VkCommandPool g_commandPool;
//
//static void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
//{
//    VkCommandBufferAllocateInfo allocInfo;
//    ZERO_OUT(allocInfo);
//    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
//    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
//    allocInfo.commandPool = g_commandPool;
//    allocInfo.commandBufferCount = 1;
//
//    VkCommandBuffer commandBuffer;
//    vkAllocateCommandBuffers(g_device, &allocInfo, &commandBuffer);
//
//    VkCommandBufferBeginInfo beginInfo;
//    ZERO_OUT(beginInfo);
//    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
//
//    vkBeginCommandBuffer(commandBuffer, &beginInfo);
//
//    VkBufferCopy copyRegion;
//    ZERO_OUT(copyRegion);
//    copyRegion.srcOffset = 0;
//    copyRegion.dstOffset = 0;
//    copyRegion.size = size;
//
//    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);
//
//    vkEndCommandBuffer(commandBuffer);
//
//    VkSubmitInfo submitInfo;
//    ZERO_OUT(submitInfo);
//    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
//    submitInfo.commandBufferCount = 1;
//    submitInfo.pCommandBuffers = &commandBuffer;
//
//    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
//    vkQueueWaitIdle(graphicsQueue);
//
//    vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
//}
//
//VertexBuffer GPU_CreateVertexBuffer(vertex_t* verts, int num_verts)
//{
//    VkDeviceSize bufferSize = sizeof(vertex_t) * num_verts;
//
//    VertexBuffer stagingBuf = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
//
//    void* data;
//    vkMapMemory(g_device, stagingBuf.memory, 0, bufferSize, 0, &data);
//    memcpy(data, verts, bufferSize);
//    vkUnmapMemory(g_device, stagingBuf.memory);
//
//    VertexBuffer vertexBuffer = createBuffer(
//        bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
//        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
//    );
//
//    copyBuffer(stagingBuf.buffer, vertexBuffer.buffer, bufferSize);
//
//    vkDestroyBuffer(g_device, stagingBuf.buffer, NULL);
//    vkFreeMemory(g_device, stagingBuf.memory, NULL);
//
//    return vertexBuffer;
//}
//
//VertexBuffer GPU_CreateIndexBuffer(uint16_t* inds, int num_indices)
//{
//    VkDevice bufferSize = sizeof(uint16_t) * num_indices;
//
//    VertexBuffer stagingBuf = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
//
//    void* data;
//    vkMapMemory(device, stagingBuf.memory, 0, bufferSize, 0, &data);
//    memcpy(data, inds, bufferSize);
//    vkUnmapMemory(device, stagingBuf.memory);
//
//    VertexBuffer indexBuffer = createBuffer(bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
//        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
//    );
//
//    copyBuffer(stagingBuf.buffer, indexBuffer.buffer, bufferSize);
//
//    vkDestroyBuffer(device, stagingBuf.buffer, NULL);
//    vkFreeMemory(device, stagingBuf.memory, NULL);
//
//    return indexBuffer;
//}
//
//void GPU_DestroyVertexBuffer(VertexBuffer* vb)
//{
//    vkDestroyBuffer(g_device, vb->buffer, NULL);
//    vkFreeMemory(g_device, vb->memory, NULL);
//}
