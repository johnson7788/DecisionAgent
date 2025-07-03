export function dictToBase64(data) {
    try {
      const jsonStr = JSON.stringify(data);
      // Use btoa for Base64 encoding (browser environment)
      return btoa(unescape(encodeURIComponent(jsonStr)));
    } catch (error) {
      console.error("Error encoding dictionary to Base64:", error);
      return null;
    }
  }
  
  export function base64ToDict(base64Str) {
    try {
      // Use atob for Base64 decoding (browser environment)
      const jsonStr = decodeURIComponent(escape(atob(base64Str)));
      return JSON.parse(jsonStr);
    } catch (error) {
      console.error("Error decoding Base64 to dictionary:", error);
      return null;
    }
  }