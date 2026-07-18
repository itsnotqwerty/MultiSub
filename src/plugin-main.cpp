#include <obs-module.h>
#include <plugin-support.h>

#include "multisub-filter.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")
OBS_MODULE_AUTHOR("MultiSub Contributors")

MODULE_EXPORT const char *obs_module_name(void)
{
	return "MultiSub";
}

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Real-time multi-modal subtitle generation for OBS";
}

MODULE_EXPORT bool obs_module_load(void)
{
	multisub::register_multisub_filter();
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

MODULE_EXPORT void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}
