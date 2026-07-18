#include "logger.h"

#include <obs-module.h>
#include <plugin-support.h>

namespace multisub {

void log_info(const std::string &message)
{
	obs_log(LOG_INFO, "[MultiSub] %s", message.c_str());
}

void log_warn(const std::string &message)
{
	obs_log(LOG_WARNING, "[MultiSub] %s", message.c_str());
}

void log_error(const std::string &message)
{
	obs_log(LOG_ERROR, "[MultiSub] %s", message.c_str());
}

} // namespace multisub
