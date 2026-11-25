using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// DanceReplayMVP - Trực quan hóa và đánh giá động tác nhảy trong Unity
/// Đề tài NCKH: Dance Movement Evaluation
/// Hỗ trợ cả 2 chế độ:
/// 1. Offline JSON Replay: Đọc file JSON đã tính toán sẵn (cực kỳ mượt và an toàn).
/// 2. Live UDP Stream: Nhận stream thời gian thực từ Python qua socket UDP.
/// </summary>
public class DanceReplayMVP : MonoBehaviour
{
    public enum PlayMode
    {
        OfflineJsonReplay,
        LiveUdpStream
    }

    [Header("=== CẤU HÌNH CHẾ ĐỘ ===")]
    public PlayMode mode = PlayMode.OfflineJsonReplay;

    [Header("=== CẤU HÌNH OFFLINE REPLAY ===")]
    [Tooltip("Kéo thả file dance_demo_data.json vào đây, hoặc để trống để đọc từ StreamingAssets")]
    public TextAsset jsonFileAsset;
    public string jsonFileName = "dance_demo_data.json";
    public bool autoPlay = true;
    public bool loop = true;
    [Range(0.2f, 2.0f)] public float playbackSpeed = 1.0f;

    [Header("=== CẤU HÌNH LIVE UDP ===")]
    public int udpPort = 5065;

    [Header("=== THIẾT LẬP HIỂN THỊ 2D / 3D ===")]
    public float skeletonScale = 4.0f;
    public Vector3 teacherOffset = new Vector3(-2.2f, -1.5f, 0f);
    public Vector3 studentOffset = new Vector3(2.2f, -1.5f, 0f);
    public float jointRadius = 0.09f;
    public float boneWidth = 0.04f;

    [Header("=== MÀU SẮC TRỰC QUAN ===")]
    public Color teacherColor = new Color(0.2f, 0.7f, 1.0f, 1.0f); // Xanh dương / Cyan
    public Color passJointColor = new Color(0.1f, 0.9f, 0.2f, 1.0f); // Xanh lá
    public Color failJointColor = new Color(0.95f, 0.2f, 0.2f, 1.0f); // Đỏ
    public Color boneColor = new Color(0.85f, 0.85f, 0.85f, 0.6f);

    // Dữ liệu nội bộ
    [System.Serializable]
    public class JointCoord
    {
        public float x;
        public float y;
        public int is_pass;
        public float joint_error;
    }

    [System.Serializable]
    public class FrameData
    {
        public int frame_index;
        public int teacher_frame_index;
        public float timestamp;
        public float frame_score;
        public float max_score = 13.0f;
        public bool is_pass;
        public float cumulative_accuracy;
        // Joint dictionaries are parsed via JSON helper
        public Dictionary<string, JointCoord> student_joints = new Dictionary<string, JointCoord>();
        public Dictionary<string, JointCoord> teacher_joints = new Dictionary<string, JointCoord>();
    }

    [System.Serializable]
    public class DemoRootData
    {
        public string project;
        public int video_id;
        public float fps;
        public int total_frames;
        public float score_threshold;
        public List<List<string>> bones;
        public List<FrameData> frames;
    }

    private DemoRootData rootData;
    private List<string[]> bonePairs = new List<string[]>();
    private int currentFrameIdx = 0;
    private float frameTimer = 0f;
    private bool isPaused = false;
    private FrameData currentFrame;

    // UDP Threading
    private Thread udpThread;
    private UdpClient udpClient;
    private string latestUdpJson = "";
    private readonly object lockObj = new object();
    private bool hasNewUdpPacket = false;

    // Visual Objects
    private class SkeletonVisual
    {
        public GameObject root;
        public Dictionary<string, GameObject> joints = new Dictionary<string, GameObject>();
        public List<LineRenderer> bones = new List<LineRenderer>();
        public List<string[]> boneJointNames = new List<string[]>();
    }

    private SkeletonVisual teacherVisual = new SkeletonVisual();
    private SkeletonVisual studentVisual = new SkeletonVisual();

    private Material jointMaterial;
    private Material boneMaterial;

    void Start()
    {
        CreateDefaultMaterials();
        InitDefaultBones();

        if (mode == PlayMode.OfflineJsonReplay)
        {
            LoadOfflineJson();
        }
        else
        {
            StartUdpListener();
        }

        CreateSkeletonVisuals();
    }

    void CreateDefaultMaterials()
    {
        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null) shader = Shader.Find("Unlit/Color");

        jointMaterial = new Material(shader);
        boneMaterial = new Material(shader);
    }

    void InitDefaultBones()
    {
        bonePairs = new List<string[]>
        {
            new string[] { "nose", "left_shoulder" },
            new string[] { "nose", "right_shoulder" },
            new string[] { "left_shoulder", "right_shoulder" },
            new string[] { "left_shoulder", "left_elbow" },
            new string[] { "left_elbow", "left_wrist" },
            new string[] { "right_shoulder", "right_elbow" },
            new string[] { "right_elbow", "right_wrist" },
            new string[] { "left_shoulder", "left_hip" },
            new string[] { "right_shoulder", "right_hip" },
            new string[] { "left_hip", "right_hip" },
            new string[] { "left_hip", "left_knee" },
            new string[] { "left_knee", "left_ankle" },
            new string[] { "right_hip", "right_knee" },
            new string[] { "right_knee", "right_ankle" }
        };
    }

    void LoadOfflineJson()
    {
        string rawJson = "";
        if (jsonFileAsset != null)
        {
            rawJson = jsonFileAsset.text;
        }
        else
        {
            string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
            if (!File.Exists(path))
            {
                // Fallback nếu chạy trong Editor
                path = Path.Combine(Application.dataPath, jsonFileName);
            }

            if (File.Exists(path))
            {
                rawJson = File.ReadAllText(path);
            }
            else
            {
                Debug.LogError($"[DanceMVP] Không tìm thấy file JSON tại: {path}");
                return;
            }
        }

        ParseJsonData(rawJson);
    }

    void ParseJsonData(string jsonText)
    {
        try
        {
            // Parse thủ công hoặc qua SimpleJSON / MiniJSON để linh hoạt với Dictionary
            rootData = ParseFullJson(jsonText);
            if (rootData != null && rootData.frames.Count > 0)
            {
                currentFrame = rootData.frames[0];
                Debug.Log($"[DanceMVP] Nạp thành công {rootData.total_frames} khung hình (Video #{rootData.video_id})");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[DanceMVP] Lỗi khi parse JSON: {ex.Message}");
        }
    }

    void CreateSkeletonVisuals()
    {
        // 1. Tạo Teacher Skeleton
        teacherVisual.root = new GameObject("Teacher_Skeleton");
        teacherVisual.root.transform.position = teacherOffset;

        // 2. Tạo Student Skeleton
        studentVisual.root = new GameObject("Student_Skeleton");
        studentVisual.root.transform.position = studentOffset;

        string[] jointNames = {
            "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        };

        SetupSkeletonHierarchy(teacherVisual, jointNames, teacherColor);
        SetupSkeletonHierarchy(studentVisual, jointNames, passJointColor);
    }

    void SetupSkeletonHierarchy(SkeletonVisual skel, string[] jointNames, Color defaultColor)
    {
        foreach (string jName in jointNames)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = jName;
            sphere.transform.parent = skel.root.transform;
            sphere.transform.localScale = Vector3.one * (jointRadius * 2f);
            
            // Xóa collider để không cản trở vật lý
            Destroy(sphere.GetComponent<Collider>());

            Renderer r = sphere.GetComponent<Renderer>();
            r.material = new Material(jointMaterial);
            r.material.color = defaultColor;

            skel.joints[jName] = sphere;
        }

        // Tạo Bones
        foreach (string[] pair in bonePairs)
        {
            GameObject boneObj = new GameObject($"Bone_{pair[0]}_{pair[1]}");
            boneObj.transform.parent = skel.root.transform;

            LineRenderer lr = boneObj.AddComponent<LineRenderer>();
            lr.material = new Material(boneMaterial);
            lr.startColor = boneColor;
            lr.endColor = boneColor;
            lr.startWidth = boneWidth;
            lr.endWidth = boneWidth;
            lr.positionCount = 2;
            lr.useWorldSpace = true;

            skel.bones.Add(lr);
            skel.boneJointNames.Add(pair);
        }
    }

    void Update()
    {
        // Phím tắt điều khiển
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isPaused = !isPaused;
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            currentFrameIdx = 0;
            frameTimer = 0f;
        }

        if (mode == PlayMode.OfflineJsonReplay)
        {
            UpdateOfflineReplay();
        }
        else
        {
            UpdateLiveUdp();
        }

        RenderFrame();
    }

    void UpdateOfflineReplay()
    {
        if (rootData == null || rootData.frames == null || rootData.frames.Count == 0 || isPaused)
            return;

        float fps = rootData.fps > 0 ? rootData.fps : 30f;
        frameTimer += Time.deltaTime * playbackSpeed;

        if (frameTimer >= (1f / fps))
        {
            frameTimer = 0f;
            currentFrameIdx++;
            if (currentFrameIdx >= rootData.frames.Count)
            {
                currentFrameIdx = loop ? 0 : rootData.frames.Count - 1;
            }
            currentFrame = rootData.frames[currentFrameIdx];
        }
    }

    void UpdateLiveUdp()
    {
        string jsonToProcess = "";
        lock (lockObj)
        {
            if (hasNewUdpPacket)
            {
                jsonToProcess = latestUdpJson;
                hasNewUdpPacket = false;
            }
        }

        if (!string.IsNullOrEmpty(jsonToProcess))
        {
            currentFrame = ParseSingleFrameJson(jsonToProcess);
        }
    }

    void RenderFrame()
    {
        if (currentFrame == null) return;

        // Cập nhật Teacher Skeleton
        UpdateSkeletonJoints(teacherVisual, currentFrame.teacher_joints, teacherOffset, false);

        // Cập nhật Student Skeleton (kèm đổi màu Đạt / Lỗi)
        UpdateSkeletonJoints(studentVisual, currentFrame.student_joints, studentOffset, true);
    }

    void UpdateSkeletonJoints(SkeletonVisual skel, Dictionary<string, JointCoord> jointsData, Vector3 baseOffset, bool isStudent)
    {
        if (jointsData == null) return;

        foreach (var kvp in jointsData)
        {
            string jName = kvp.Key;
            JointCoord coord = kvp.Value;

            if (skel.joints.ContainsKey(jName))
            {
                GameObject obj = skel.joints[jName];
                // Chuyển tọa độ [0, 1] sang World Space có tâm ở giữa
                Vector3 worldPos = baseOffset + new Vector3(
                    (coord.x - 0.5f) * skeletonScale,
                    (coord.y - 0.5f) * skeletonScale,
                    0f
                );
                obj.transform.position = worldPos;

                if (isStudent)
                {
                    Renderer r = obj.GetComponent<Renderer>();
                    if (r != null)
                    {
                        r.material.color = (coord.is_pass == 1) ? passJointColor : failJointColor;
                    }
                }
            }
        }

        // Cập nhật các đường xương
        for (int i = 0; i < skel.bones.Count; i++)
        {
            LineRenderer lr = skel.bones[i];
            string[] pair = skel.boneJointNames[i];

            if (skel.joints.ContainsKey(pair[0]) && skel.joints.ContainsKey(pair[1]))
            {
                Vector3 p0 = skel.joints[pair[0]].transform.position;
                Vector3 p1 = skel.joints[pair[1]].transform.position;
                lr.SetPosition(0, p0);
                lr.SetPosition(1, p1);
            }
        }
    }

    void OnGUI()
    {
        if (currentFrame == null) return;

        // HUD Dashboard phong cách Cyberpunk / Dark UI
        GUI.color = Color.white;
        int boxWidth = 360;
        int boxHeight = 220;
        Rect hudRect = new Rect(20, 20, boxWidth, boxHeight);
        GUI.Box(hudRect, "DANCE EVALUATION SYSTEM - AI MVP");

        GUILayout.BeginArea(new Rect(35, 50, boxWidth - 30, boxHeight - 40));

        GUILayout.Label($"<b>Khung hình:</b> {currentFrame.frame_index} / {(rootData != null ? rootData.total_frames : 0)} ({currentFrame.timestamp:F2}s)");
        
        // Điểm số khung hình
        GUIStyle scoreStyle = new GUIStyle(GUI.skin.label);
        scoreStyle.fontSize = 16;
        scoreStyle.fontStyle = FontStyle.Bold;
        GUILayout.Label($"<b>Điểm khung hình:</b> {currentFrame.frame_score:F1} / 13.0", scoreStyle);

        // Badge PASS / FAIL
        GUIStyle badgeStyle = new GUIStyle(GUI.skin.button);
        badgeStyle.fontSize = 14;
        badgeStyle.fontStyle = FontStyle.Bold;

        if (currentFrame.is_pass)
        {
            GUI.backgroundColor = new Color(0.1f, 0.8f, 0.2f);
            GUILayout.Button("TRẠNG THÁI: PASS (ĐẠT)", badgeStyle, GUILayout.Height(30));
        }
        else
        {
            GUI.backgroundColor = new Color(0.9f, 0.2f, 0.2f);
            GUILayout.Button("TRẠNG THÁI: FAIL (LỖI ĐỘNG TÁC)", badgeStyle, GUILayout.Height(30));
        }

        GUI.backgroundColor = Color.white;
        GUILayout.Space(5);
        GUILayout.Label($"<b>Độ chính xác tích lũy:</b> {currentFrame.cumulative_accuracy:F1}%");

        // Điều khiển
        GUILayout.Space(5);
        GUILayout.Label("[Space]: Tạm dừng/Phát | [R]: Xem lại từ đầu");

        GUILayout.EndArea();

        // Nhãn ghi chú dưới 2 nhân vật
        GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.fontSize = 16;
        labelStyle.fontStyle = FontStyle.Bold;
        labelStyle.alignment = TextAnchor.MiddleCenter;

        Vector3 screenTeacher = Camera.main.WorldToScreenPoint(teacherOffset + Vector3.down * 1.5f);
        Vector3 screenStudent = Camera.main.WorldToScreenPoint(studentOffset + Vector3.down * 1.5f);

        GUI.color = teacherColor;
        GUI.Label(new Rect(screenTeacher.x - 100, Screen.height - screenTeacher.y, 200, 30), "GIÁO VIÊN (MẪU CHUẨN)", labelStyle);

        GUI.color = Color.white;
        GUI.Label(new Rect(screenStudent.x - 100, Screen.height - screenStudent.y, 200, 30), "HỌC VIÊN (AI ĐÁNH GIÁ)", labelStyle);
    }

    void StartUdpListener()
    {
        udpThread = new Thread(new ThreadStart(UdpListenWorker));
        udpThread.IsBackground = true;
        udpThread.Start();
        Debug.Log($"[DanceMVP] Đang lắng nghe UDP port {udpPort}...");
    }

    void UdpListenWorker()
    {
        udpClient = new UdpClient(udpPort);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                lock (lockObj)
                {
                    latestUdpJson = text;
                    hasNewUdpPacket = true;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[DanceMVP] Socket: {ex.Message}");
                break;
            }
        }
    }

    void OnDestroy()
    {
        if (udpThread != null && udpThread.IsAlive) udpThread.Abort();
        if (udpClient != null) udpClient.Close();
    }

    // ==========================================
    // JSON PARSER TỰ VIẾT (Không phụ thuộc plugin ngoài)
    // ==========================================
    DemoRootData ParseFullJson(string json)
    {
        DemoRootData data = new DemoRootData();
        data.frames = new List<FrameData>();

        // Lấy thông tin cơ bản
        data.project = ExtractString(json, "project");
        data.fps = ExtractFloat(json, "fps", 30f);
        data.total_frames = ExtractInt(json, "total_frames", 0);
        data.score_threshold = ExtractFloat(json, "score_threshold", 7.1f);

        // Tách các frame block
        int framesIdx = json.IndexOf("\"frames\":");
        if (framesIdx < 0) return data;

        int startBracket = json.IndexOf('[', framesIdx);
        int endBracket = json.LastIndexOf(']');
        if (startBracket < 0 || endBracket < 0) return data;

        string framesArrayStr = json.Substring(startBracket + 1, endBracket - startBracket - 1);
        List<string> frameBlocks = SplitJsonObjects(framesArrayStr);

        foreach (string block in frameBlocks)
        {
            FrameData fd = ParseSingleFrameJson(block);
            if (fd != null) data.frames.Add(fd);
        }

        return data;
    }

    FrameData ParseSingleFrameJson(string block)
    {
        FrameData fd = new FrameData();
        fd.frame_index = ExtractInt(block, "frame_index", 0);
        fd.teacher_frame_index = ExtractInt(block, "teacher_frame_index", 0);
        fd.timestamp = ExtractFloat(block, "timestamp", 0f);
        fd.frame_score = ExtractFloat(block, "frame_score", 0f);
        fd.max_score = ExtractFloat(block, "max_score", 13f);
        fd.is_pass = block.Contains("\"is_pass\": true");
        fd.cumulative_accuracy = ExtractFloat(block, "cumulative_accuracy", 0f);

        fd.student_joints = ParseJointsDict(block, "student_joints");
        fd.teacher_joints = ParseJointsDict(block, "teacher_joints");

        return fd;
    }

    Dictionary<string, JointCoord> ParseJointsDict(string parentJson, string dictKey)
    {
        Dictionary<string, JointCoord> dict = new Dictionary<string, JointCoord>();
        int keyPos = parentJson.IndexOf($"\"{dictKey}\":");
        if (keyPos < 0) return dict;

        int openBrace = parentJson.IndexOf('{', keyPos);
        if (openBrace < 0) return dict;

        int closeBrace = FindClosingBrace(parentJson, openBrace);
        if (closeBrace < 0) return dict;

        string sub = parentJson.Substring(openBrace + 1, closeBrace - openBrace - 1);
        string[] jointNames = {
            "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        };

        foreach (string jName in jointNames)
        {
            int jPos = sub.IndexOf($"\"{jName}\":");
            if (jPos >= 0)
            {
                int jOpen = sub.IndexOf('{', jPos);
                int jClose = FindClosingBrace(sub, jOpen);
                if (jOpen >= 0 && jClose > jOpen)
                {
                    string jBlock = sub.Substring(jOpen, jClose - jOpen + 1);
                    JointCoord jc = new JointCoord();
                    jc.x = ExtractFloat(jBlock, "x", 0.5f);
                    jc.y = ExtractFloat(jBlock, "y", 0.5f);
                    jc.is_pass = ExtractInt(jBlock, "is_pass", 1);
                    jc.joint_error = ExtractFloat(jBlock, "joint_error", 0f);
                    dict[jName] = jc;
                }
            }
        }

        return dict;
    }

    int FindClosingBrace(string text, int openPos)
    {
        int depth = 0;
        for (int i = openPos; i < text.Length; i++)
        {
            if (text[i] == '{') depth++;
            else if (text[i] == '}')
            {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    List<string> SplitJsonObjects(string arrayContent)
    {
        List<string> objects = new List<string>();
        int depth = 0;
        int start = -1;

        for (int i = 0; i < arrayContent.Length; i++)
        {
            char c = arrayContent[i];
            if (c == '{')
            {
                if (depth == 0) start = i;
                depth++;
            }
            else if (c == '}')
            {
                depth--;
                if (depth == 0 && start >= 0)
                {
                    objects.Add(arrayContent.Substring(start, i - start + 1));
                    start = -1;
                }
            }
        }
        return objects;
    }

    float ExtractFloat(string json, string key, float defaultVal)
    {
        string pattern = $"\"{key}\":";
        int idx = json.IndexOf(pattern);
        if (idx < 0) return defaultVal;
        int valStart = idx + pattern.Length;
        int valEnd = json.IndexOfAny(new char[] { ',', '}', '\n', '\r' }, valStart);
        if (valEnd < 0) valEnd = json.Length;
        string numStr = json.Substring(valStart, valEnd - valStart).Trim();
        if (float.TryParse(numStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float res))
            return res;
        return defaultVal;
    }

    int ExtractInt(string json, string key, int defaultVal)
    {
        return (int)ExtractFloat(json, key, defaultVal);
    }

    string ExtractString(string json, string key)
    {
        string pattern = $"\"{key}\":";
        int idx = json.IndexOf(pattern);
        if (idx < 0) return "";
        int valStart = json.IndexOf('"', idx + pattern.Length);
        if (valStart < 0) return "";
        int valEnd = json.IndexOf('"', valStart + 1);
        if (valEnd < 0) return "";
        return json.Substring(valStart + 1, valEnd - valStart - 1);
    }
}
